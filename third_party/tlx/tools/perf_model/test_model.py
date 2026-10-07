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


def test_unknown_queue_capacity_is_partial_and_blocks_scheduling():
    model = Model(completion_credit_graph(None))
    report = model.bounds(target_period=4)
    assert report["bound_is_partial"]
    assert report["queues"][0]["capacity"] is None
    assert report["queue_lifetimes"][0]["minimum_credits_at_target"] == 5
    with pytest.raises(Unknown, match="queue capacities"):
        model.schedule()


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


def test_partition_placement_preserves_transport_and_compute_work():
    first = make_model(block_k=128, array_mapping="striped")
    second = make_model(block_k=128, array_mapping="striped", wave_partitions=[0, 1, 0, 1])
    for key in ("wmma_instructions", "unique_input_bytes", "lds_payload_read_bytes", "lds_scale_read_bytes",
                "lds_instructions"):
        assert first["workload"][key] == second["workload"][key]
    assert [op for op in first["operations"] if op["phase"] == "epilogue"] == [
        op for op in second["operations"] if op["phase"] == "epilogue"
    ]
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
