"""CPU regressions for resource, lifetime, and measurement contracts."""

import json
from pathlib import Path

import pytest

from third_party.tlx.tools.perf_model.model import Model, Unknown, latency_statistics, window_metrics
from third_party.tlx.tools.perf_model.mxfp import make_model

HERE = Path(__file__).parent


def event(schedule, name, iteration=0):
    return next(e for e in schedule["events"] if e["op"] == name and e["iteration"] == iteration)


def completion_credit_graph(capacity=1):
    return dict(
        parameters=dict(credits=dict(value=capacity)),
        resources=dict(issue=dict(capacity=1), matrix=dict(capacity=1)),
        compute_resources=["matrix"],
        operations=[
            dict(id="issue", kind="lds", latency=1, uses=[dict(resource="issue", work=1)], domain="issue"),
            dict(id="array", latency=2),
            dict(id="done", latency=17),
            dict(id="mma", kind="mma", uses=[dict(resource="matrix", work=1)]),
        ],
        dependencies=[dict(source=a, target=b) for a, b in (("issue", "array"), ("array", "done"), ("done", "mma"))],
        queues=[
            dict(id="outstanding", capacity="credits", scope="wave0", unit="instructions",
                 entries=[dict(acquire="issue", release="done")])
        ],
        requests=[
            dict(id="read", issue="issue", stages=["array"], complete="done", source_release="array", consumers=["mma"],
                 kind="lds")
        ],
    )


@pytest.mark.parametrize("credits,period", [(1, 20), (2, 10), (4, 5)])
def test_completion_credits_survive_source_release(credits, period):
    model = Model(completion_credit_graph(credits))
    assert model.bounds()["period_lower_bound"] == period
    schedule = model.schedule(iterations=8, warmup=2)
    read = schedule["requests"]["events"][0]
    assert read["source_release"] == 3
    assert read["complete"] == 20
    assert event(schedule, "issue", credits)["start"] >= 20
    queue = schedule["queue_occupancy"][0]
    assert queue["peak_credits"] <= credits
    assert queue["credit_ticks"] == 8 * 20


@pytest.mark.parametrize("ordered", [False, True])
def test_unknown_queue_capacity_is_partial_and_blocks_scheduling(ordered):
    graph = completion_credit_graph(None)
    graph["queues"][0]["ordered_retirement"] = ordered
    model = Model(graph)
    report = model.bounds(target_period=4)
    assert report["bound_is_partial"]
    assert report["queues"][0]["capacity"] is None
    assert report["queue_lifetimes"][0]["minimum_credits_at_target"] == 5
    with pytest.raises(Unknown, match="queue capacities"):
        model.schedule()


@pytest.mark.parametrize("release_delay", [None, 7])
@pytest.mark.parametrize("explicit_retirement", [False, True])
def test_ordered_credits_wait_for_older_completion_without_serializing_transport(release_delay, explicit_retirement):
    entries = [dict(acquire=f"issue{i}", release=f"ready{i}") for i in range(2)]
    if release_delay is not None:
        entries[0]["release_delay"] = release_delay
    graph = dict(
        resources=dict(issue=dict(capacity=1)),
        operations=[dict(id=f"issue{i}", uses=[dict(resource="issue", work=1)]) for i in range(2)] +
        [dict(id="ready0", latency=20), dict(id="ready1", latency=1)],
        dependencies=[dict(source=f"issue{i}", target=f"ready{i}") for i in range(2)],
        queues=[dict(id="loads", capacity=2, ordered_retirement=True, entries=entries)],
    )
    if explicit_retirement:
        for i, entry in enumerate(entries):
            entry["retire"] = f"retire{i}"
            graph["operations"].append(dict(id=entry["retire"]))
    model = Model(graph)
    schedule = model.schedule(iterations=3, warmup=0)
    old_ready = event(schedule, "ready0")["start"] + (20 if release_delay is None else release_delay)
    young_ready = event(schedule, "ready1")["end"]
    assert young_ready < old_ready
    # A younger result can be ready while both instruction credits remain
    # occupied. The next pair cannot issue using that younger credit early.
    for i in range(2):
        retired = model.queues[0]["entries"][i]["release"]
        assert event(schedule, retired)["start"] == old_ready
        assert event(schedule, f"issue{i}", 1)["start"] >= old_ready
    # Normalizing a queue must not rewrite the caller's release contract.
    assert graph["queues"][0]["entries"][0]["release"] == "ready0"


def test_ordered_retirement_includes_the_iteration_boundary():
    graph = dict(
        resources=dict(issue=dict(capacity=1)),
        operations=[
            dict(id="issue", uses=[dict(resource="issue", work=1)]),
            dict(id="fast", latency=1),
            dict(id="slow", latency=20),
            dict(id="next")
        ],
        dependencies=[dict(source="issue", target=name)
                      for name in ("fast", "next")] + [dict(source="next", target="slow")],
        queues=[
            dict(id="loads", capacity=10, ordered_retirement=True,
                 entries=[dict(acquire="issue", release="fast"),
                          dict(acquire="next", release="slow")])
        ],
    )
    model = Model(graph)
    schedule = model.schedule(iterations=3, warmup=0)
    younger = event(schedule, "fast", 1)["end"]
    older = event(schedule, "slow")["end"]
    assert younger < older
    retired = model.queues[0]["entries"][0]["release"]
    assert event(schedule, retired, 1)["start"] >= older


def test_shared_array_delays_an_already_issued_request():
    graph = dict(
        resources=dict(array=dict(capacity=1), matrix=dict(capacity=1)),
        compute_resources=["matrix"],
        operations=[
            dict(id="issue", latency=1),
            dict(id="array", uses=[dict(resource="array", work=4)]),
            dict(id="done", latency=2),
            dict(id="mma", uses=[dict(resource="matrix", work=1)]),
            dict(id="copy", uses=[dict(resource="array", work=8)]),
        ],
        dependencies=[dict(source=a, target=b) for a, b in (("issue", "array"), ("array", "done"), ("done", "mma"))],
        requests=[dict(id="load", issue="issue", stages=["array"], complete="done", consumers=["mma"], kind="lds")],
    )
    schedule = Model(graph).schedule(iterations=1, warmup=0)
    assert event(schedule, "issue")["start"] == 0
    assert event(schedule, "array")["start"] == 8
    read = schedule["requests"]["events"][0]
    assert read["latency"] == 14
    assert read["route_queue_ticks"] == 7


@pytest.mark.parametrize("stages,aggregate,queued", [([], False, None), (["array"], True, None), (["array"], False, 0)])
def test_request_queueing_requires_a_complete_nonaggregate_route(stages, aggregate, queued):
    graph = completion_credit_graph()
    graph["requests"][0].update(stages=stages, aggregate=aggregate)
    schedule = Model(graph).schedule(iterations=1, warmup=0)
    read = schedule["requests"]["events"][0]
    assert read["latency"] == 20
    assert read["route_queue_ticks"] == queued
    assert read["route_complete"] == (queued is not None)
    for requests in (schedule["requests"]["by_kind"], schedule["windows"]["dispatch"]["requests"]):
        assert requests["lds"]["latency"]["mean"] == 20
        assert requests["lds"]["route_queue_ticks"]["mean"] == queued
        assert requests["lds"]["route_queue_ticks"]["count"] == (queued is not None)


def test_request_dependency_delay_is_transport_not_queueing():
    graph = dict(resources={}, operations=[dict(id="issue", latency=2),
                                           dict(id="done", latency=1)],
                 dependencies=[dict(source="issue", target="done", delay=delay) for delay in (4, 9, 6)],
                 requests=[dict(id="read", issue="issue", complete="done")])
    schedule = Model(graph).schedule(iterations=1, warmup=0)
    read = schedule["requests"]["events"][0]
    assert read["latency"] == 10
    assert read["route_queue_ticks"] == 0


def test_first_covering_wait_can_precede_the_real_consumer():
    graph = dict(
        resources=dict(matrix=dict(capacity=1)),
        compute_resources=["matrix"],
        operations=[
            dict(id="issue"),
            dict(id="old", latency=5),
            dict(id="young", latency=20),
            dict(id="arrival", latency=8),
            dict(id="gate"),
            dict(id="later", latency=100),
            dict(id="mma", uses=[dict(resource="matrix", work=1)])
        ],
        dependencies=[dict(source="issue", target=name) for name in ("old", "young", "arrival", "later")] +
        [dict(source=name, target="mma") for name in ("young", "gate", "later")],
        waits=[dict(id="merged-counter-wait", arrival="arrival", target="gate", completions=["old", "young"])],
        requests=[dict(id="younger", issue="issue", complete="young", consumers=["mma"], kind="lds")],
    )
    schedule = Model(graph).schedule(iterations=1, warmup=0)
    wait = schedule["waits"][0]
    assert wait["arrival"] == 8 and wait["ready"] == 20
    assert wait["completion_wait_ticks"] == 12
    read = schedule["requests"]["events"][0]
    assert read["lead_to_first_wait"] == 8
    assert read["first_consumer_start"] == 100
    assert read["completion_after_first_wait"] == 12


def test_workgroup_join_reports_the_slow_peer_once():
    graph = dict(
        resources={}, operations=[
            dict(id="fast", latency=4),
            dict(id="slow", latency=10),
            dict(id="arrive", latency=5),
            dict(id="barrier", latency=2)
        ], waits=[dict(id="visibility", arrival="arrive", target="barrier", completions=["fast", "slow"])])
    schedule = Model(graph).schedule(iterations=1, warmup=0)
    wait = schedule["waits"][0]
    assert wait["completion_wait_ticks"] == 5
    assert wait["last_completions"] == [dict(op="slow", iteration=0)]
    assert schedule["makespan"] == 12


def test_partner_waves_and_overlapping_stall_categories():
    report = window_metrics(0, 12, {"simd0": [(0, 4), (4, 8)], "simd1": [(2, 6)]},
                            {"simd0": {"credit": [(6, 11)], "barrier": [(9, 12)]}})
    assert report["service_ticks"] == 12
    assert report["available_resource_ticks"] == 24
    assert report["service_fraction"] == 0.5
    row = report["resources"]["simd0"]
    assert row["exposed_by_category"] == dict(credit=3, barrier=3)
    assert row["exposed_union_ticks"] == 4
    assert row["unattributed_idle_ticks"] == 0


def test_latency_bins_use_sample_weights_and_zero_means_unknown():
    report = latency_statistics(
        [dict(count=1, sum=10, minimum=10, maximum=10),
         dict(count=3, sum=90, minimum=20, maximum=40)])
    assert report == dict(count=4, sum=100, mean=25, minimum=10, maximum=40)
    assert latency_statistics([dict(count=0, sum=0, minimum=1000000, maximum=0)])["mean"] is None
    assert latency_statistics([dict(count=1, sum=10), dict(count=1, sum=20, maximum=20)])["maximum"] is None


@pytest.mark.parametrize("slots,period", [(1, 20), (2, 10), (3, 8)])
def test_streaming_period_still_has_a_verified_wrap(slots, period):
    graph = json.loads((HERE / "streaming.json").read_text())
    model = Model(graph, overrides=dict(slots=slots))
    assert model.bounds()["period_lower_bound"] == period
    schedule = model.schedule(iterations=20, warmup=4)
    assert schedule["periodic_witness"]["cycles_per_iteration"] == period
    assert schedule["buffer_lifetimes"][0]["reuse_slack"]["minimum"] >= 0


def test_code_footprint_is_not_dynamic_work_or_a_fetch_latency():
    graph = json.loads((HERE / "streaming.json").read_text())
    original = Model(graph).schedule(iterations=8, warmup=2)
    graph["code"] = dict(total_bytes=16000, tail_bytes=8000)
    annotated = Model(graph).schedule(iterations=8, warmup=2)
    assert annotated["makespan"] == original["makespan"]
    graph["operations"].append(dict(id="cold_tail_fetch", kind="fetch", latency=11, phase="epilogue"))
    timed = Model(graph).schedule(iterations=8, warmup=2)
    assert timed["makespan"] == original["makespan"] + 11
    assert timed["compute_service_ticks"] == original["compute_service_ticks"]
    assert Model(graph).bounds()["period_lower_bound"] == 8


def test_used_register_count_does_not_override_the_allocation():
    graph = dict(
        resources={}, operations=[dict(id="work")], storage=dict(vgpr=dict(fixed=498, allocated=528, capacity=1024)),
        residency=[
            dict(resource="vgpr", per_cta=528, capacity=1024),
            dict(resource="lds", per_cta=65536, capacity=None)
        ])
    report = Model(graph).bounds()
    assert report["storage"]["vgpr"]["allocated"] == 528
    assert report["residency"]["upper_bound_from_known_constraints"] == 1
    assert not report["residency"]["all_constraints_resolved"]
    graph["residency"][0]["per_cta"] = 512
    assert Model(graph).bounds()["residency"]["upper_bound_from_known_constraints"] == 2


def test_coexecution_empty_slot_does_not_claim_operand_readiness():
    graph = dict(
        resources=dict(issue=dict(capacity=1), matrix=dict(capacity=1)), compute_resources=["matrix"], operations=[
            dict(id="compute", kind="matrix", window_domain="issue", uses=[dict(resource="matrix", work=8)]),
            dict(id="operand_ready", latency=12),
            dict(id="load", kind="lds", domain="issue", uses=[dict(resource="issue", work=1)])
        ], dependencies=[dict(source="operand_ready", target="load")], coexecution=dict(matrix=[["lds"]] * 8))
    schedule = Model(graph).schedule(iterations=1, warmup=0)
    assert schedule["issue_windows"][0]["kinds"]["lds"]["empty_permitted_ticks"] == 8
    assert event(schedule, "load")["start"] == 12


@pytest.mark.parametrize("transport", ["fixed", "staged"])
@pytest.mark.parametrize("weight_bits,reads", [(8, 272), (4, 208)])
def test_mxfp_native_work_and_instruction_credits(transport, weight_bits, reads):
    graph = make_model(block_k=128, weight_bits=weight_bits, transport=transport)
    profile = json.loads((HERE / "hypothetical.json").read_text())
    model = Model(graph, profile)
    report = model.bounds()
    assert graph["workload"]["wmma_instructions"] == 256
    assert graph["workload"]["lds_instructions"] == reads
    assert report["compute_service_cycles"] == 512
    pending = [q for q in report["queues"] if q["id"].startswith("lds_pending")]
    assert [q["credits_per_phase"] for q in pending] == [reads // 4] * 4
    assert all(q["unit"] == "instructions" and q["scope"].startswith("wave") for q in pending)


def test_mxfp_staged_schedule_reports_retirement_and_full_drain():
    graph = make_model(tile_m=64, tile_n=64, block_k=128)
    profile = json.loads((HERE / "hypothetical.json").read_text())
    schedule = Model(graph, profile).schedule(iterations=4, warmup=1, policy="deadline")
    assert schedule["verified"]
    assert schedule["requests"]["by_kind"]["lds"]["route_queue_ticks"]["maximum"] > 0
    assert all(row["reuse_slack"]["minimum"] >= 0 for row in schedule["buffer_lifetimes"])
    assert schedule["phase_windows"]["epilogue"][1] == schedule["makespan"]
    assert len(schedule["waits"]) == 4
    requests = {(row["request"], row["iteration"]): row for row in schedule["requests"]["events"]}
    for queue in graph["queues"]:
        if queue["id"].startswith("lds_pending"):
            completions = [
                requests[entry["acquire"], iteration]["complete"]
                for iteration in range(4)
                for entry in queue["entries"]
            ]
            assert completions == sorted(completions)


@pytest.mark.parametrize("scale_read_bytes,last_chunk", [(128, 3), (256, 1), (512, 0)])
def test_mxfp_matrix_waits_only_for_its_packed_scale_fragment(scale_read_bytes, last_chunk):
    graph = make_model(block_k=256, scale_read_bytes=scale_read_bytes)
    requests = {request["id"]: request for request in graph["requests"]}
    # C00 can start before scales used by the opposite edge of the wave tile.
    # A scales follow M; B scales follow N, independently in each K128 step.
    for step in (0, 1):
        for operand, far, near in (("A", "7_0", "0_7"), ("B", "0_7", "7_0")):
            first = requests[f"scale_{operand}_0_k{step}_w0"]["consumers"]
            last = requests[f"scale_{operand}_{last_chunk}_k{step}_w0"]["consumers"]
            assert f"mma_issue_k{step}_w0_0_0" in first
            assert f"mma_issue_k{step}_w0_{near}" in first
            assert f"mma_issue_k{step}_w0_{far}" in last
            assert (f"mma_issue_k{step}_w0_{far}" in first) == (last_chunk == 0)
            assert not any(f"_k{1 - step}_" in consumer for consumer in first + last)


def test_partition_placement_preserves_transport_and_compute_work():
    first = make_model(block_k=128, array_mapping="striped")
    second = make_model(block_k=128, array_mapping="striped", wave_partitions=[0, 1, 0, 1])
    for key in ("wmma_instructions", "unique_input_bytes", "lds_payload_read_bytes", "lds_scale_read_bytes",
                "lds_instructions"):
        assert first["workload"][key] == second["workload"][key]
    assert [op for op in first["operations"]
            if op["phase"] == "epilogue"] == [op for op in second["operations"] if op["phase"] == "epilogue"]
    profile = json.loads((HERE / "hypothetical.json").read_text())
    assert Model(first, profile).bounds()["compute_service_cycles"] == Model(second,
                                                                             profile).bounds()["compute_service_cycles"]


@pytest.mark.parametrize("weight_bits,lds_bytes,reads", [(8, 198 * 1024, 408), (4, 134 * 1024, 280)])
def test_eight_waves_preserve_math_and_increase_operand_replication(weight_bits, lds_bytes, reads):
    graph = make_model(block_k=128, weight_bits=weight_bits, waves_m=4, waves_n=2)
    profile = json.loads((HERE / "hypothetical.json").read_text())
    assert graph["workload"]["lds_payload_read_bytes"] + graph["workload"]["lds_scale_read_bytes"] == lds_bytes
    assert graph["workload"]["lds_instructions"] == reads
    assert Model(graph, profile).bounds()["compute_service_cycles"] == 512
    assert len([q for q in graph["queues"] if q["id"].startswith("lds_pending")]) == 8


def test_loaded_observations_do_not_become_component_delays():
    graph = completion_credit_graph()
    report = Model(graph, dict(parameters={}, observations=[dict(metric="loaded latency", mean=500,
                                                                 scope="CU0")])).bounds()
    assert report["period_lower_bound"] == 20
    assert report["observations"][0]["mean"] == 500


def test_request_stages_must_be_connected():
    graph = completion_credit_graph()
    graph["dependencies"] = [edge for edge in graph["dependencies"] if edge["target"] != "array"]
    graph["queues"] = []
    with pytest.raises(ValueError, match="route order"):
        Model(graph)


def test_a_shorter_wait_does_not_imply_less_total_idle():
    old = window_metrics(0, 20, {"simd": [(0, 8)]}, {"simd": {"tensor": [(8, 16)], "barrier": [(16, 20)]}})
    new = window_metrics(0, 20, {"simd": [(0, 8)]}, {"simd": {"tensor": [(8, 10)], "barrier": [(10, 20)]}})
    assert new["resources"]["simd"]["exposed_by_category"]["tensor"] < old["resources"]["simd"]["exposed_by_category"][
        "tensor"]
    assert new["resources"]["simd"]["exposed_union_ticks"] == old["resources"]["simd"]["exposed_union_ticks"]
    assert new["service_fraction"] == old["service_fraction"]


def test_stage_credit_is_held_while_downstream_service_is_blocked():
    graph = dict(
        resources=dict(array=dict(capacity=1)),
        operations=[dict(id="busy", uses=[dict(resource="array", work=10)]),
                    dict(id="decode", latency=1)] +
        [
            op for i in range(2)
            for op in (dict(id=f"stage{i}", latency=1), dict(id=f"array{i}", uses=[dict(resource="array", work=4)]))
        ],
        dependencies=[
            edge for i in range(2)
            for edge in (dict(source="decode", target=f"stage{i}"), dict(source=f"stage{i}", target=f"array{i}"))
        ],
        queues=[
            dict(id="stage", capacity=1,
                 entries=[dict(acquire=f"stage{i}", release=f"array{i}", release_delay=0) for i in range(2)])
        ],
    )
    schedule = Model(graph).schedule(iterations=1, warmup=0)
    assert event(schedule, "stage0")["start"] == 1
    assert event(schedule, "array0")["start"] == 10
    assert event(schedule, "stage1")["start"] == 10
    assert event(schedule, "stage1")["start"] < event(schedule, "array0")["end"]
    queue = schedule["queue_occupancy"][0]
    assert queue["maximum_lifetime"] == 9
    assert queue["credit_ticks"] == 13
    assert schedule["admissions"][1]["credit_wait_ticks"] == 9


@pytest.mark.parametrize("independent_work,wait_ticks", [(4, 16), (24, 0)])
def test_loaded_latency_and_full_queue_do_not_imply_exposed_wait(independent_work, wait_ticks):
    graph = completion_credit_graph()
    graph["operations"] += [
        dict(id="independent", uses=[dict(resource="matrix", work=independent_work)]),
        dict(id="arrival"),
        dict(id="gate")
    ]
    graph["dependencies"] += [dict(source="independent", target="arrival"), dict(source="gate", target="mma")]
    graph["requests"][0]["compute_resource"] = "matrix"
    graph["waits"] = [
        dict(id="input", arrival="arrival", target="gate", completions=["done"], compute_resources=["matrix"])
    ]
    schedule = Model(graph).schedule(iterations=1, warmup=0, analysis_windows=dict(early=[0, 8], late=[16, 21]))
    dispatch = schedule["windows"]["dispatch"]
    queue = dispatch["queues"][0]
    assert queue["saturated_ticks"] == 20
    assert queue["credit_wait_ticks"] == 0  # No second admission is waiting.
    assert dispatch["requests"]["lds"]["latency"]["mean"] == 20
    assert schedule["waits"][0]["completion_wait_ticks"] == wait_ticks
    assert dispatch["matrix_resources"]["matrix"]["idle_overlap_union_ticks"] == wait_ticks
    early, late = schedule["windows"]["early"], schedule["windows"]["late"]
    assert early["requests"]["lds"]["latency"]["mean"] is None
    assert late["requests"]["lds"]["latency"]["mean"] == 20  # Keep pre-window time in the sample.
    assert late["queues"][0]["credit_ticks"] == 4
    assert late["queues"][0]["acquisitions"] == 0
    assert late["queues"][0]["releases"] == 1
    if not wait_ticks:
        assert dispatch["service_fraction"] == 1


def arbitration_graph(policy="round-robin", capacity=1):
    return dict(
        resources=dict(array=dict(
            capacity=capacity, arbitration=dict(policy=policy, classes=[
                dict(kind="read", weight=2), dict(kind="copy", weight=1)
            ]))),
        operations=[dict(id=f"read{i}", kind="read", uses=[dict(resource="array", work=1, rate=1)]) for i in range(5)] +
        [dict(id=f"copy{i}", kind="copy", uses=[dict(resource="array", work=3, rate=1)]) for i in range(2)],
    )


def test_weighted_arbitration_counts_grants_and_preserves_class_fifo():
    model = Model(arbitration_graph())
    schedule = model.schedule(iterations=1, warmup=0)
    grants = schedule["arbitration"]["array"]["grants"]
    assert [(row["op"], row["start"]) for row in grants] == [("read0", 0), ("read1", 1), ("copy0", 2), ("read2", 5),
                                                             ("read3", 6), ("copy1", 7), ("read4", 10)]
    assert schedule["makespan"] == 11
    assert schedule["periodic_witness"] is None
    assert "stateful arbitration" in schedule["periodic_witness_limit"]


def test_arbitration_skips_unready_classes_without_idling_the_port():
    graph = arbitration_graph()
    graph["operations"].append(dict(id="late", latency=4))
    graph["dependencies"] = [dict(source="late", target=f"copy{i}") for i in range(2)]
    schedule = Model(graph).schedule(iterations=1, warmup=0)
    grants = schedule["arbitration"]["array"]["grants"]
    assert [(row["op"], row["start"]) for row in grants[:5]] == [("read0", 0), ("read1", 1), ("read2", 2), ("read3", 3),
                                                                 ("copy0", 4)]
    assert schedule["makespan"] == 11


def test_priority_arbitration_and_pooled_grants_do_not_count_zero_time_attempts():
    schedule = Model(arbitration_graph("priority", capacity=2)).schedule(iterations=1, warmup=0)
    grants = schedule["arbitration"]["array"]["grants"]
    assert [row["op"] for row in grants[:5]] == [f"read{i}" for i in range(5)]
    assert [row["start"] for row in grants[:4]] == [0, 0, 1, 1]
    for blocked in schedule["scheduling_blocks"]:
        started = event(schedule, blocked["op"])["start"]
        for spans in blocked["reasons"].values():
            assert all(begin < end <= started for begin, end in spans)


@pytest.mark.parametrize("weight", [0, -1, 1.5])
def test_arbitration_rejects_invalid_grant_weights(weight):
    graph = arbitration_graph()
    graph["resources"]["array"]["arbitration"]["classes"][0]["weight"] = weight
    with pytest.raises(ValueError, match="positive integer"):
        Model(graph)


def test_mxfp_stage_queues_preserve_work_and_require_an_explicit_capacity():
    graph = make_model(tile_m=64, tile_n=64, block_k=128, stage_queues=True, array_arbitration="round-robin")
    original = make_model(tile_m=64, tile_n=64, block_k=128)
    for key in ("wmma_instructions", "unique_input_bytes", "lds_instructions", "array_bytes_per_period"):
        assert graph["workload"][key] == original["workload"][key]
    profile = json.loads((HERE / "hypothetical.json").read_text())
    model = Model(graph, profile)
    schedule = model.schedule(iterations=4, warmup=1, policy="deadline")
    assert schedule["verified"]
    assert schedule["windows"]["interior"]["service_fraction"] <= 1
    stages = {q["queue"]: q for q in schedule["queue_occupancy"]}
    assert stages["tdm_copy_fifo"]["peak_credits"] <= 4
    for group in (0, 1):
        assert stages[f"ds_stage{group}"]["maximum_lifetime"] >= profile["parameters"]["ds_scheduler_native"]
    del profile["parameters"]["tdm_copy_fifo_depth"]
    model = Model(graph, profile)
    assert "tdm_copy_fifo" in model.bounds()["unresolved_queues"]
    with pytest.raises(Unknown, match="queue capacities"):
        model.schedule(iterations=4, warmup=1)


def test_mxfp_fifo_residence_is_independent_of_total_return_latency():
    graph = make_model(tile_m=64, tile_n=64, block_k=128, stage_queues=True)
    request = next(r for r in graph["requests"] if r["id"] == "input_A_0")
    route = {request["issue"], *request["stages"], request["complete"]}
    fifo = next(q for q in graph["queues"] if q["id"] == "tdm_copy_fifo")
    isolated = dict(parameters=graph["parameters"], resources=graph["resources"],
                    operations=[op for op in graph["operations"] if op["id"] in route],
                    dependencies=[e for e in graph["dependencies"] if e["source"] in route and e["target"] in route],
                    queues=[dict(fifo, entries=[e for e in fifo["entries"]
                                                if e["acquire"] in route])], requests=[dict(request, consumers=[])])
    profile = json.loads((HERE / "hypothetical.json").read_text())
    latencies = []
    for residence in (0, 8):
        schedule = Model(isolated, profile, dict(tdm_copy_fifo_cycles=residence)).schedule(iterations=1, warmup=0)
        read = schedule["requests"]["events"][0]
        latencies.append(read["latency"])
        assert read["route_queue_ticks"] == 0
        assert schedule["queue_occupancy"][0]["maximum_lifetime"] == residence
    assert latencies[0] == latencies[1]
    with pytest.raises(ValueError, match="time cannot be negative"):
        Model(isolated, profile, dict(tdm_copy_fifo_cycles=9)).schedule(iterations=1, warmup=0)


def test_short_array_requests_still_occupy_whole_ports():
    graph = make_model(tile_m=64, tile_n=64, block_k=128, packet_bytes=128)
    graph = dict(parameters=graph["parameters"],
                 resources={name: graph["resources"][name]
                            for name in ("array_pool", "array_ports")},
                 operations=[op for op in graph["operations"] if op["kind"] == "tdm_array"][:3])
    profile = json.loads((HERE / "hypothetical.json").read_text())
    schedule = Model(graph, profile).schedule(iterations=1, warmup=0)
    # Two 256-byte ports cannot serve three independent 128-byte requests
    # in one clock, even though their combined byte capacity would suffice.
    assert [e["start"] for e in schedule["events"]] == [0, 0, 1]
    assert schedule["makespan"] == 2
