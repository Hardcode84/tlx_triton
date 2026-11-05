"""Build an MXFP resource graph before choosing an assembly or TLX schedule.

A period transfers BK128 or BK256 and computes one or two native K128 steps.
Time is in shader cycles. Machine calibration is supplied separately.
"""

import argparse
import json
from pathlib import Path


def make_model(tile_m=256, tile_n=256, waves_m=2, waves_n=2, weight_bits=8, block_k=256, data_slots=2, scale_slots=3,
               register_slots=1, scale_register_slots=2, packet_bytes=256, memory_path="cache", scalar_ops=8,
               vector_ops=0, a_register_slots=None, b_register_slots=None, mma_order="mn", array_mapping="pooled",
               lds_release="workgroup", transport="staged", scale_read_bytes=256, lds_order="interleaved",
               wave_partitions=None, stage_queues=False, array_arbitration="scheduler"):
    if weight_bits not in (4, 8) or block_k not in (128, 256):
        raise ValueError("require A8W8/A8W4 and BK128/BK256")
    if min(tile_m, tile_n, waves_m, waves_n, data_slots, scale_slots, register_slots, scale_register_slots,
           packet_bytes) <= 0:
        raise ValueError("dimensions, packet size and ring depths must be positive")
    if tile_m % (16 * waves_m) or tile_n % (16 * waves_n) or packet_bytes % 128:
        raise ValueError("wave tiles must be multiples of 16x16; packets must be multiples of 128 bytes")
    if memory_path not in ("cache", "memory") or scalar_ops < 0 or vector_ops < 0:
        raise ValueError("invalid memory path or instruction count")
    if mma_order not in ("mn", "nm") or any(n is not None and n < 1 for n in (a_register_slots, b_register_slots)):
        raise ValueError("require positive operand register depths and mn/nm matrix order")
    if array_mapping not in ("pooled", "striped"):
        raise ValueError("require pooled or striped array port mapping")
    if lds_release not in ("source", "completion", "workgroup"):
        raise ValueError("require source, completion, or workgroup LDS release")
    if transport not in ("staged", "fixed") or lds_order not in ("operand", "interleaved"):
        raise ValueError("require staged/fixed transport and operand/interleaved LDS admission")
    if array_arbitration not in ("scheduler", "round-robin", "tdm-priority", "lds-priority"):
        raise ValueError("invalid array arbitration policy")
    if transport != "staged" and (stage_queues or array_arbitration != "scheduler"):
        raise ValueError("stage queues and array arbitration require staged transport")
    if scale_read_bytes not in (128, 256, 512):
        raise ValueError("scale reads must return 128, 256, or 512 bytes per wave instruction")
    waves = waves_m * waves_n
    if waves not in (4, 8):
        raise ValueError("this topology has four SIMDs and supports four or eight waves")
    if wave_partitions is None:
        wave_partitions = [(wave % 4) // 2 for wave in range(waves)]
    if len(wave_partitions) != waves or any(port not in (0, 1) for port in wave_partitions):
        raise ValueError("wave_partitions must select port 0 or 1 for each wave")
    if array_mapping == "pooled" and wave_partitions != [(wave % 4) // 2 for wave in range(waves)]:
        raise ValueError("wave partition placement requires striped array mapping")
    wave_m, wave_n = tile_m // waves_m, tile_n // waves_n
    substeps = block_k // 128
    parameters = {}

    def param(name, value, unit, source, kind="structural"):
        parameters[name] = dict(value=value, unit=unit, source=source, kind=kind)

    # Raw stage delays are kept separate from their inferred route composition.
    # A bindings profile supplies values and replaces their provenance.
    raw = {
        "shader_period_ps": "ps", "gfx_period_ps": "ps", "fabric_period_ps": "ps", "memory_period_ps": "ps",
        "ds_decode_native": "shader cycles", "ds_scheduler_native": "gfx cycles", "ds_ram_native": "gfx cycles",
        "ds_read_output_native": "gfx cycles", "ds_return_native": "gfx cycles", "sp_bypass_native": "shader cycles",
        "ta_native": "gfx cycles", "tcp_input_native": "gfx cycles", "tcp_output_native": "gfx cycles",
        "l1_input_native": "fabric cycles", "tcx_request_native": "fabric cycles", "tcx_return_native": "fabric cycles",
        "df_request_native": "fabric cycles", "df_return_native": "fabric cycles", "cache_input_native":
        "fabric cycles", "cache_output_native": "fabric cycles", "mc_read_request_native": "memory cycles",
        "mc_read_return_native": "memory cycles", "mc_write_request_native": "memory cycles", "mc_write_return_native":
        "memory cycles", "dram_activate_native": "memory cycles", "dram_sequence_native": "memory cycles",
        "memory_cycles_per_32bytes": "memory cycles / 32 bytes / channel", "array_port_width":
        "bytes / gfx cycle / port", "return_link_width": "bytes / gfx cycle / DS group", "store_link_width":
        "bytes / gfx cycle / DS group", "cache_link_width": "bytes / fabric cycle / CU", "memory_channels":
        "channels in modeled scope", "active_cus": "CUs sharing those channels", "barrier_cycles": "shader cycles",
        "tdm_command_ii": "shader cycles / logical command", "address_vgprs": "VGPRs / lane / wave", "lds_padding":
        "allocated bytes / logical byte", "lds_outstanding": "instructions / wave, held through result completion",
        "lds_residency_capacity": "aggregate LDS bytes available to resident CTAs / CU", "tdm_copy_fifo_depth":
        "returned packets / CU, held through array admission", "tdm_copy_fifo_cycles":
        "shader cycles of intrinsic return-pipe delay while holding a FIFO entry"
    }
    for name, unit in raw.items():
        param(name, None, unit, "unfilled: supply a calibration or explicit scenario", "unfilled")
    for name, value, unit, source in (
        ("mma_cycles", 8, "shader cycles / instruction", "LLVM: A8W8 and A8W4 16x16x128 scaled WMMA service"),
        ("mma_scale_lead", 1, "shader cycles", "assumed issue-to-execution scale-read lead"),
        ("mma_payload_hold", 8, "shader cycles from execution", "assumed last source read at end of matrix service"),
        ("mma_scale_hold", 1, "shader cycles from issue", "assumed scale source-read lifetime"),
        ("mma_result_latency", 8, "shader cycles from execution", "initial accumulator dependency latency"),
        ("mma_store_latency", 24, "shader cycles from issue", "LLVM gfx1250 WMMA-to-DS RAW hazard"),
        ("valu_memory_latency", 16, "shader cycles", "LLVM gfx1250 non-WMMA VALU-to-DS RAW hazard"),
        ("issue_rate", 1, "instructions / shader cycle / SIMD", "initial single instruction issue domain"),
        ("mma_pending", 2, "instructions / SIMD", "assumed one running plus one pending matrix instruction"),
        ("ds_scheduler_depth", 2, "entries / DS group", "initial scheduler occupancy scenario"),
        ("tdm_descriptors", 6, "descriptors / issuing SIMD", "initial outstanding-descriptor scenario"),
        ("transfer_credits", 256, "sectors / CU", "initial outstanding-transfer scenario"),
        ("transfer_sector_bytes", 128, "bytes / sector", "transaction credit granularity"),
        ("array_lds_weight", 1, "stage grants / arbitration quantum", "chosen LDS request arbitration scenario"),
        ("array_tdm_weight", 1, "stage grants / arbitration quantum", "chosen TDM request arbitration scenario"),
        ("return_width_multiplier", 1, "ratio", "sensitivity axis for return-interface interpretation"),
        ("vgprs_per_simd", 1024, "VGPRs / lane / SIMD", "register capacity scenario"),
        ("vgpr_granularity", 16, "VGPRs / lane / wave", "allocation granularity scenario"),
        ("lds_limit", 320 * 1024, "bytes / CU", "maximum LDS capacity scenario"),
        ("lds_granularity", 256, "bytes", "allocation granularity scenario"),
        ("data_slots", data_slots, "physical BK tiles", "chosen payload ring"),
        ("scale_slots", scale_slots, "physical BK tiles", "chosen independent scale ring"),
        ("register_slots", register_slots, "K128 fragments", "chosen per-fragment payload register ring"),
        ("a_register_slots", a_register_slots if a_register_slots is not None else "register_slots", "K128 fragments",
         "A register ring; defaults to shared payload depth"),
        ("b_register_slots", b_register_slots if b_register_slots is not None else "register_slots", "K128 fragments",
         "B register ring; defaults to shared payload depth"),
        ("scale_register_slots", scale_register_slots, "K128 scale sets", "chosen packed-scale register ring"),
        ("address_slots", 1, "physical BK address sets / wave", "chosen address-register lifetime depth"),
    ):
        param(name, value, unit, source)

    def derived(name, expression, unit="shader cycles"):
        param(name, expression, unit, "explicit route or capacity composition; see assumptions", "derived")

    for name in ("ds_scheduler", "ds_ram", "ds_read_output", "ds_return", "ta", "tcp_input", "tcp_output"):
        derived(name + "_cycles", f"ceil({name}_native * gfx_period_ps / shader_period_ps)")
    for name in ("l1_input", "tcx_request", "tcx_return", "df_request", "df_return", "cache_input", "cache_output"):
        derived(name + "_cycles", f"ceil({name}_native * fabric_period_ps / shader_period_ps)")
    for name in ("mc_read_request", "mc_read_return", "mc_write_request", "mc_write_return", "dram_activate",
                 "dram_sequence"):
        derived(name + "_cycles", f"ceil({name}_native * memory_period_ps / shader_period_ps)")
    derived("array_bw", "array_port_width * shader_period_ps / gfx_period_ps", "bytes / shader cycle / port")
    derived("return_bw", "return_link_width * return_width_multiplier * shader_period_ps / gfx_period_ps",
            "bytes / shader cycle / DS group")
    derived("store_bw", "store_link_width * shader_period_ps / gfx_period_ps", "bytes / shader cycle / DS group")
    derived("cache_link_bw", "cache_link_width * shader_period_ps / fabric_period_ps", "bytes / shader cycle / CU")
    derived("memory_bw",
            "memory_channels * 32 / memory_cycles_per_32bytes / active_cus * shader_period_ps / memory_period_ps",
            "bytes / shader cycle / CU")
    derived("tdm_command_rate", "1 / tdm_command_ii", "logical commands / shader cycle / CU")
    derived("ds_array_offset", "ds_decode_native + ds_scheduler_cycles")
    derived(
        "cache_request_cycles", "ta_cycles + tcp_input_cycles + l1_input_cycles + tcx_request_cycles + "
        "df_request_cycles + cache_input_cycles")
    derived("cache_return_cycles", "cache_output_cycles + df_return_cycles + tcx_return_cycles")
    derived("read_miss_request_cycles", "mc_read_request_cycles + dram_activate_cycles + dram_sequence_cycles")
    derived("read_miss_return_cycles", "mc_read_return_cycles")
    derived("write_ack_cycles", "cache_return_cycles + tcp_output_cycles")

    resources = {
        "cache_read_link": dict(capacity="cache_link_bw", unit="bytes/cycle", role="input data transport"),
        "cache_write_link": dict(capacity="cache_link_bw", unit="bytes/cycle", role="output data transport"),
        "tdm_issue": dict(capacity="tdm_command_rate", unit="logical commands/cycle")
    }
    if memory_path == "memory":
        resources["memory_channels"] = dict(capacity="memory_bw", unit="bytes/cycle",
                                            role="read/write bandwidth share; no inter-CTA cache reuse")
    if array_mapping == "pooled":
        resources["array_pool"] = dict(capacity="2 * array_bw", unit="bytes/cycle",
                                       role="two dynamically allocated array ports; each request uses at most one")
        resources["array_ports"] = dict(capacity=2, unit="occupied ports",
                                        role="one whole port per request during array service, including short packets")
    for group in range(2):
        if array_mapping == "striped":
            resources[f"array{group}"] = dict(capacity="array_bw", unit="bytes/cycle",
                                              role="shared cache/LDS array read and write port")
            resources[f"array_slot{group}"] = dict(capacity=1, unit="occupied ports")
        resources[f"lds_return{group}"] = dict(capacity="return_bw", unit="bytes/cycle",
                                               role="LDS-to-register return; excludes TDM")
        resources[f"lds_store{group}"] = dict(capacity="store_bw", unit="bytes/cycle",
                                              role="register-to-LDS ingress; excludes TDM")
        resources[f"ds_scheduler{group}"] = dict(capacity="ds_scheduler_depth", unit="occupied entries",
                                                 role="short DS scheduler stage, not full return latency")
    for simd in range(4):
        resources[f"xdl{simd}"] = dict(capacity=1, unit="service cycles/cycle")
        resources[f"issue{simd}"] = dict(capacity="issue_rate", unit="instructions/cycle")
    if array_arbitration != "scheduler":
        classes = [dict(kind="lds_array", weight="array_lds_weight"), dict(kind="tdm_array", weight="array_tdm_weight")]
        if array_arbitration == "tdm-priority":
            classes.reverse()
        for resource in (["array_pool"] if array_mapping == "pooled" else ["array0", "array1"]):
            resources[resource]["arbitration"] = dict(
                policy="round-robin" if array_arbitration == "round-robin" else "priority", classes=classes)
    operations, dependencies, buffers, queues, requests, waits = [], [], [], [], [], []
    requests_by_completion = {}
    lds_entries = {wave: [] for wave in range(waves)}
    scheduler_entries = {group: [] for group in range(2)}
    copy_fifo_entries = {}

    def op(name, kind, latency=0, uses=(), domain=None, phase="loop", window_domain=None):
        operations.append(
            dict(id=name, kind=kind, latency=latency, uses=list(uses), domain=domain, phase=phase,
                 window_domain=window_domain))
        return name

    def use(resource, work, offset=0, rate=None):
        item = dict(resource=resource, work=work, offset=offset)
        if rate is not None:
            item["rate"] = rate
        return item

    def array_uses(work, port, offset=0):
        return [
            use("array_pool" if array_mapping == "pooled" else f"array{port}", work, offset, "array_bw"),
            use("array_ports" if array_mapping == "pooled" else f"array_slot{port}", f"ceil(({work}) / array_bw)",
                offset, 1)
        ]

    def edge(src, dst, distance=0, delay=None, reason="dependency"):
        item = dict(source=src, target=dst, distance=distance, reason=reason)
        if delay is not None:
            item["delay"] = delay
        dependencies.append(item)

    def queue(name, capacity, entries, role, scope="CU", unit="credits", ordered_retirement=False):
        queues.append(
            dict(id=name, capacity=capacity, entries=entries, role=role, scope=scope, unit=unit,
                 ordered_retirement=ordered_retirement))

    def entry(acquire, release, units=1, release_delay=None):
        item = dict(acquire=acquire, release=release, units=units)
        if release_delay is not None:
            item["release_delay"] = release_delay
        return item

    def sync(name, phase="loop"):
        return op(name, "control", "barrier_cycles", [use(f"issue{s}", waves // 4) for s in range(4)], phase=phase)

    def request(name, issue, complete, stages, kind, scope, amount, **extra):
        item = dict(id=name, issue=issue, complete=complete, stages=stages, kind=kind, scope=scope, bytes=amount,
                    consumers=[], **extra)
        requests.append(item)
        requests_by_completion[complete] = item

    # Initial address setup is outside K. Further non-memory work is explicit,
    # one instruction per node, so it can fit nonconsecutive coexecution slots.
    for wave in range(waves):
        issue = f"issue{wave % 4}"
        op(f"setup_w{wave}", "valu", "valu_memory_latency", [use(issue, 2)], issue, "prologue")

    sizes = {"S": (tile_m + tile_n) * block_k // 32, "A": tile_m * block_k, "B": tile_n * block_k * weight_bits // 8}
    commands, visibility, packets, readers = {}, {}, {}, {name: [] for name in sizes}
    descriptor_entries = []
    for name, size in sizes.items():
        commands[name] = op(f"command_{name}", "tdm", uses=[use("issue0", 1), use("tdm_issue", 1)], domain="issue0")
        complete = op(f"complete_{name}", "completion")
        visibility[name] = sync(f"visible_{name}")
        edge(complete, visibility[name])
        request(f"command_{name}", commands[name], complete, [], "tdm_command", "SIMD0", size,
                descriptors=2 if name == "S" else 1, aggregate=True)
        requests_by_completion[complete]["consumers"].append(visibility[name])
        # Partial fusion combines the two scale descriptors into one command.
        descriptor_entries.append(entry(commands[name], complete, 2 if name == "S" else 1))
        packets[name] = []
        for index, offset in enumerate(range(0, size, packet_bytes)):
            amount = min(packet_bytes, size - offset)
            port = index % 2
            link_offset = "cache_request_cycles + cache_return_cycles"
            reservations = []
            if memory_path == "memory":
                reservations.append(use("memory_channels", amount, "cache_request_cycles + read_miss_request_cycles"))
                link_offset += f" + read_miss_request_cycles + ceil({amount} / memory_bw) + read_miss_return_cycles"
            reservations.append(use("cache_read_link", amount, link_offset))
            array_offset = f"{link_offset} + ceil({amount} / cache_link_bw) + tcp_output_cycles"
            reservations.extend(array_uses(amount, port, array_offset))
            packet_name = f"input_{name}_{index}"
            stages = []
            if transport == "staged":
                packet = op(packet_name, "tdm_packet", "cache_request_cycles")
                previous = packet
                if memory_path == "memory":
                    memory = op(packet_name + "_memory", "tdm_memory",
                                f"read_miss_request_cycles + ceil({amount} / memory_bw) + read_miss_return_cycles",
                                [use("memory_channels", amount, "read_miss_request_cycles")])
                    edge(previous, memory)
                    stages.append(memory)
                    previous = memory
                returned = op(
                    packet_name + "_return", "tdm_transport", f"cache_return_cycles + ceil({amount} / cache_link_bw)" +
                    ("" if stage_queues else " + tcp_output_cycles"),
                    [use("cache_read_link", amount, "cache_return_cycles")])
                edge(previous, returned)
                stages.append(returned)
                finished = op(packet_name + "_array", "tdm_array", uses=array_uses(amount, port))
                if stage_queues:
                    transit = op(packet_name + "_pre_fifo", "tdm_stage", "tcp_output_cycles - tdm_copy_fifo_cycles")
                    admitted = op(packet_name + "_copy_fifo", "tdm_stage", "tdm_copy_fifo_cycles")
                    edge(returned, transit)
                    edge(transit, admitted)
                    edge(admitted, finished)
                    stages.extend([transit, admitted])
                    copy_fifo_entries[packet] = entry(admitted, finished, release_delay=0)
                else:
                    edge(returned, finished)
            else:
                packet = op(packet_name, "tdm_packet", uses=reservations)
                finished = packet
            edge(commands[name], packet)
            edge(finished, complete)
            request(packet_name, packet, finished, stages, "tdm_input", "CU", amount)
            requests_by_completion[finished]["consumers"].append(visibility[name])
            packets[name].append((packet, finished, amount))
    queue("input_descriptors", "tdm_descriptors", descriptor_entries,
          "one issuing SIMD; scale command consumes two descriptor credits", scope="SIMD0", unit="descriptors")
    # Model an interleaving policy explicitly. Admission order is not inferred
    # from the order in which Python happens to construct the resource nodes.
    packet_entries = [
        entry(packet, finished, f"ceil({amount} / transfer_sector_bytes)")
        for index in range(max(map(len, packets.values())))
        for stream in packets.values() if index < len(stream) for packet, finished, amount in [stream[index]]
    ]
    queue("input_transactions", "transfer_credits", packet_entries,
          "outstanding sectors, from admission through final array write", unit="sectors")
    if stage_queues:
        queue("tdm_copy_fifo", "tdm_copy_fifo_depth", [copy_fifo_entries[e["acquire"]] for e in packet_entries],
              "returned packets wait here until the array accepts them; upstream sectors remain outstanding",
              unit="packets")

    mma = {}
    payload_registers, scale_registers = {}, {}
    address_acquire, address_release = {}, {}
    for wave in range(waves):
        address_acquire[wave] = op(f"address_acquire_w{wave}", "register_acquire")
        address_release[wave] = op(f"address_release_w{wave}", "register_release")

    def lds_read(name, amount, wave, step, operand, index=0, chunk=0):
        issue, group = f"issue{wave % 4}", (wave % 4) // 2
        hold = f"ds_array_offset + ceil({amount} / array_bw) + ds_ram_cycles"
        return_offset = f"{hold} + ds_read_output_cycles + ds_return_cycles"
        ready = f"{return_offset} + ceil({amount} / return_bw) + sp_bypass_native"
        stages = []
        if transport == "staged":
            node = op(name, "lds", "ds_decode_native", [use(issue, 1)], issue)
            scheduler = op(name + "_scheduler", "lds_stage", "ds_scheduler_cycles",
                           [] if stage_queues else [use(f"ds_scheduler{group}", "ds_scheduler_cycles", rate=1)])
            array = op(name + "_array", "lds_array", f"ceil({amount} / array_bw) + ds_ram_cycles",
                       array_uses(amount, wave_partitions[wave]))
            returned = op(name + "_return", "lds_return",
                          f"ds_read_output_cycles + ds_return_cycles + ceil({amount} / return_bw) + sp_bypass_native",
                          [use(f"lds_return{group}", amount, "ds_read_output_cycles + ds_return_cycles")])
            edge(node, scheduler)
            edge(scheduler, array)
            edge(array, returned)
            complete = op(name + "_complete", "completion")
            edge(returned, complete)
            stages = [scheduler, array, returned]
            source = dict(op=array)
        else:
            node = op(name, "lds", ready, [
                use(issue, 1),
                use(f"ds_scheduler{group}", "ds_scheduler_cycles", "ds_decode_native", rate=1),
                *array_uses(amount, wave_partitions[wave], "ds_array_offset"),
                use(f"lds_return{group}", amount, return_offset)
            ], issue)
            returned, source = node, dict(op=node, delay=hold)
            complete = op(name + "_complete", "completion")
            edge(returned, complete)
        edge(node, address_release[wave], delay="ds_decode_native", reason="address source read")
        request(name, node, complete, stages, "lds", f"wave{wave}", amount, source_release=source,
                compute_resource=f"xdl{wave % 4}")
        order = ((index, operand, chunk) if lds_order == "interleaved" else (operand, index, chunk))
        if stage_queues:
            scheduler_entries[group].append(
                ((step, not name.startswith("scale_"), *order, wave), entry(scheduler, array, release_delay=0)))
        # Array/return work may finish independently. Counter retirement and
        # consumers must also wait for earlier same-wave LDS instructions.
        pending = dict(entry(node, returned), retire=complete)
        lds_entries[wave].append(((step, not name.startswith("scale_"), *order), pending))
        return node, complete, source

    for step in range(substeps):
        for wave in range(waves):
            issue, simd = f"issue{wave % 4}", wave % 4
            address = op(f"address_ready_k{step}_w{wave}", "address")
            previous = None
            for index in range(scalar_ops if step == 0 else 0):
                work = op(f"scalar{index}_w{wave}", "scalar", uses=[use(issue, 1)], domain=issue)
                if previous:
                    edge(previous, work)
                else:
                    edge(address_acquire[wave], work)
                previous = work
            for index in range(vector_ops):
                work = op(f"vector{index}_k{step}_w{wave}", "valu", uses=[use(issue, 1)], domain=issue)
                if previous:
                    edge(previous, work)
                else:
                    edge(address_acquire[wave], work)
                previous = work
            if previous:
                edge(previous, address, delay="valu_memory_latency" if vector_ops else None)
            else:
                edge(address_acquire[wave] if step == 0 else f"address_ready_k0_w{wave}", address, delay=0)
            fragments, scales, consumed_by = {}, {}, {}
            for name, rows in (("A", wave_m), ("B", wave_n)):
                # Packed scale bytes are real traffic; two small register sets
                # permit prefetch without doubling all payload registers.
                loads = []
                acquire = op(f"scale_acquire_{name}_k{step}_w{wave}", "register_acquire")
                for index, offset in enumerate(range(0, rows * 4, scale_read_bytes)):
                    issued, load, source = lds_read(f"scale_{name}_{index}_k{step}_w{wave}",
                                                    min(scale_read_bytes, rows * 4 - offset), wave, step, name, index)
                    edge(acquire, issued)
                    edge(visibility["S"], issued)
                    edge(address, issued)
                    readers["S"].append((issued, load, source))
                    loads.append(load)
                scales[name] = loads
                release = op(f"scale_release_{name}_k{step}_w{wave}", "register_release")
                scale_registers.setdefault((wave, name), []).append(entry(acquire, release))
            for name, count, bits in (("A", wave_m // 16, 8), ("B", wave_n // 16, weight_bits)):
                for index in range(count):
                    fragment = f"{name}{index}_k{step}_w{wave}"
                    acquire = op(f"acquire_{fragment}", "register_acquire")
                    loads = []
                    for chunk in range(16 * 128 * bits // 8 // 512):
                        issued, load, source = lds_read(f"read_{fragment}_{chunk}", 512, wave, step, name, index, chunk)
                        edge(acquire, issued)
                        edge(visibility[name], issued)
                        edge(address, issued)
                        readers[name].append((issued, load, source))
                        loads.append(load)
                    fragments[name, index] = loads
                    release = op(f"release_{fragment}", "register_release")
                    consumed_by[name, index] = release
                    payload_registers.setdefault((wave, name, index), []).append(entry(acquire, release))
            for m in range(wave_m // 16):
                for n in range(wave_n // 16):
                    suffix = f"k{step}_w{wave}_{m}_{n}"
                    issue_mma = op(f"mma_issue_{suffix}", "mma", "mma_scale_lead", [use(issue, 1)], issue)
                    execute = op(f"mma_execute_{suffix}", "mma_execute", "mma_result_latency",
                                 [use(f"xdl{simd}", "mma_cycles")], window_domain=issue)
                    edge(issue_mma, execute)
                    mma[step, wave, m, n] = (issue_mma, execute)
                    for name, index in (("A", m), ("B", n)):
                        # One native operand uses four scale bytes for each
                        # of its 16 rows. Scales for other M/N fragments can
                        # still be in flight when this instruction starts.
                        scale_chunk = index * 16 * 4 // scale_read_bytes
                        for load in fragments[name, index] + [scales[name][scale_chunk]]:
                            edge(load, issue_mma)
                            requests_by_completion[load]["consumers"].append(issue_mma)
                        edge(execute, consumed_by[name, index], delay="mma_payload_hold")
                        edge(issue_mma, f"scale_release_{name}_k{step}_w{wave}", delay="mma_scale_hold")
            control = op(f"loop_control_k{step}_w{wave}", "control", uses=[use(issue, 1)], domain=issue)
            edge(mma[step, wave, wave_m // 16 - 1, wave_n // 16 - 1][0], control)
    for wave in range(waves):
        queue(f"lds_pending_w{wave}", "lds_outstanding", [e for _, e in sorted(lds_entries[wave])],
              "LDS instruction credits held from issue through result completion; separate from DS scheduler entries",
              scope=f"wave{wave}", unit="instructions", ordered_retirement=True)
        queue(f"address_register_w{wave}", "address_slots", [entry(address_acquire[wave], address_release[wave])],
              "address set remains live through all DS source reads in a physical BK tile", scope=f"wave{wave}",
              unit="address sets")
        for step in range(substeps):
            following = (step + 1) % substeps
            edge(f"loop_control_k{step}_w{wave}", mma[following, wave, 0, 0][0], int(following == 0),
                 reason="loop control before next compute step")
    if stage_queues:
        for group, entries in scheduler_entries.items():
            queue(f"ds_stage{group}", "ds_scheduler_depth", [e for _, e in sorted(entries)],
                  "decoded requests hold scheduler entries until array admission, including downstream stalls",
                  scope=f"DS group{group}", unit="instructions")
    for key, entries in payload_registers.items():
        queue("payload_register_" + "_".join(map(str, key)), key[1].lower() + "_register_slots", entries,
              "per-fragment overwrite after every last payload source read; depth measured in K128 steps",
              scope=f"wave{key[0]}", unit="K128 fragments")
    for key, entries in scale_registers.items():
        queue("scale_register_" + "_".join(map(str, key)), "scale_register_slots", entries,
              "packed scales released after issue-time scale read", scope=f"wave{key[0]}", unit="scale sets")

    # Pick a realizable matrix order; other work remains free to move. Waves on
    # each SIMD alternate instructions. This is a design choice, not a hardware
    # arbitration claim. Pending instructions cannot create compute service.
    for simd in range(4):
        coordinates = [(m, n) for m in range(wave_m // 16) for n in range(wave_n // 16)]
        if mma_order == "nm":
            coordinates.sort(key=lambda pair: pair[::-1])
        ordered = [
            mma[step, wave, m, n] for step in range(substeps) for m, n in coordinates for wave in range(simd, waves, 4)
        ]
        queue(f"matrix_pending{simd}", "mma_pending", [entry(i, e, release_delay="mma_cycles") for i, e in ordered],
              "running plus pending matrix instructions on one SIMD", scope=f"SIMD{simd}", unit="instructions")
        for index, (_, execute) in enumerate(ordered):
            following = ordered[(index + 1) % len(ordered)][1]
            edge(execute, following, int(index + 1 == len(ordered)), "mma_cycles", "matrix execution order")
    for step, wave, m, n in mma:
        following = (step + 1) % substeps
        edge(mma[step, wave, m, n][1], mma[following, wave, m, n][1], int(following == 0), "mma_result_latency",
             "accumulator recurrence")

    if lds_release == "workgroup":
        # A full LDS wait cannot release one operand independently while the
        # other operands/scales from this stage still return to registers.
        released_all = sync("released_all")
        arrived_all = op("reuse_arrival_all", "arrival")
        completions = []
        for operand_readers in readers.values():
            for issued, load, _ in operand_readers:
                edge(issued, arrived_all, delay=1, reason="all reads issued before reuse wait")
                completions.append(load)
        waits.append(
            dict(id="reuse_all", arrival=arrived_all, target=released_all, completions=completions,
                 compute_resources=[f"xdl{s}" for s in range(4)], role="full-stage workgroup LDS completion"))
    for name, size in sizes.items():
        if lds_release == "workgroup":
            released = released_all
        else:
            released = op(f"released_{name}", "buffer_release", "barrier_cycles")
            for _, load, source in readers[name]:
                # Source release is an optimistic hardware-lifetime bound.
                # Completion retains the slot until its result has returned.
                edge(source["op"] if lds_release == "source" else load, released,
                     delay=source.get("delay") if lds_release == "source" else None,
                     reason="LDS last source read" if lds_release == "source" else "operand LDS results complete")
        buffers.append(
            dict(id=f"lds_{name}", producer=commands[name], consumers=[released],
                 size=f"ceil({size} * lds_padding / lds_granularity) * lds_granularity",
                 slots="scale_slots" if name == "S" else "data_slots", space="lds"))

    # Native FP32 accumulator stores, two b128 instructions per 16x16 result.
    # A single staged output tile reuses the input LDS after the K loop drains.
    output_bytes = tile_m * tile_n * 4
    output_visible = sync("output_visible", "epilogue")
    for wave in range(waves):
        issue, group = f"issue{wave % 4}", (wave % 4) // 2
        for m in range(wave_m // 16):
            for n in range(wave_n // 16):
                for part in range(2):
                    array_offset = "ds_array_offset + ceil(512 / store_bw)"
                    store = op(
                        f"output_b128_w{wave}_{m}_{n}_{part}", "lds_store", uses=[
                            use(issue, 1),
                            use(f"ds_scheduler{group}", "ds_scheduler_cycles", "ds_decode_native", rate=1),
                            use(f"lds_store{group}", 512, "ds_array_offset"), *array_uses(512, group, array_offset)
                        ], domain=issue, phase="epilogue")
                    edge(mma[substeps - 1, wave, m, n][0], store, delay="mma_store_latency", reason="WMMA-to-DS RAW")
                    edge(store, output_visible)
    output_command = op("output_command", "tdm", uses=[use("issue0", 1), use("tdm_issue", 1)], domain="issue0",
                        phase="epilogue")
    edge(output_visible, output_command)
    output_complete = op("output_complete", "completion", phase="epilogue")
    output_entries = []
    for index, offset in enumerate(range(0, output_bytes, packet_bytes)):
        amount = min(packet_bytes, output_bytes - offset)
        link_offset = f"ceil({amount} / array_bw) + tcp_output_cycles"
        uses = array_uses(amount, index % 2) + [use("cache_write_link", amount, link_offset)]
        completion = f"{link_offset} + ceil({amount} / cache_link_bw) + cache_request_cycles + write_ack_cycles"
        if memory_path == "memory":
            memory_offset = f"{link_offset} + ceil({amount} / cache_link_bw) + cache_request_cycles + mc_write_request_cycles"
            uses.append(use("memory_channels", amount, memory_offset))
            completion += f" + mc_write_request_cycles + ceil({amount} / memory_bw) + mc_write_return_cycles"
        packet = op(f"output_packet_{index}", "tdm_packet", completion, uses, phase="epilogue")
        edge(output_command, packet)
        edge(packet, output_complete)
        output_entries.append(entry(packet, packet, f"ceil({amount} / transfer_sector_bytes)"))
    queue("output_transactions", "transfer_credits", output_entries, "output sectors through modeled acknowledgment",
          unit="sectors")

    accumulators = tile_m * tile_n // (32 * waves)
    a_vgprs = wave_m * 128 // (32 * 4)
    b_vgprs = wave_n * 128 * weight_bits // 8 // (32 * 4)
    scale_vgprs = (wave_m + wave_n) * 4 / (32 * 4)
    vgprs = (f"{accumulators} + {a_vgprs} * a_register_slots + {b_vgprs} * b_register_slots + "
             f"{scale_vgprs} * scale_register_slots + address_vgprs * address_slots")
    param("vgpr_next_free", vgprs, "VGPRs / lane / wave",
          "working-set estimate; replace with the compiled descriptor, including occupancy reservations", "derived")
    derived("vgpr_allocated_per_wave", "ceil(vgpr_next_free / vgpr_granularity) * vgpr_granularity",
            "VGPRs / lane / wave")
    storage = {
        "lds":
        dict(capacity="lds_limit",
             phase_fixed=dict(epilogue=f"ceil({output_bytes} * lds_padding / lds_granularity) * lds_granularity"))
    }
    lds_per_cta = "max(" + " + ".join(f"({b['size']}) * {b['slots']}"
                                      for b in buffers) + ", " + storage["lds"]["phase_fixed"]["epilogue"] + ")"
    residency = [
        dict(resource="lds", capacity="lds_residency_capacity", per_cta=lds_per_cta, scope="CU",
             source="separate aggregate admission capacity; lds_limit is only a per-workgroup launch limit")
    ]
    for simd in range(4):
        allocated = f"{len(range(simd, waves, 4))} * vgpr_allocated_per_wave"
        storage[f"vgpr_simd{simd}"] = dict(
            capacity="vgprs_per_simd", unit="VGPRs/lane/SIMD",
            fixed=f"{len(range(simd, waves, 4))} * ceil(({vgprs}) / vgpr_granularity) * vgpr_granularity",
            allocated=allocated, allocation_basis="rounded next_free_vgpr; estimate until descriptor calibration")
        residency.append(
            dict(resource=f"vgpr_simd{simd}", capacity="vgprs_per_simd", per_cta=allocated, scope=f"SIMD{simd}"))
    # LLVM's scaled eight-cycle pattern. The first scale-read issue is a
    # separate node, so the S slot can admit the next instruction at offset 7
    # without incorrectly turning eight cycles of matrix service into seven.
    classes = {
        "0": ["control"], "E": ["control", "scalar", "lds", "lds_store",
                                "tdm"], "I": ["control", "scalar", "lds", "lds_store", "tdm", "valu"], "S":
        ["control", "scalar", "lds", "lds_store", "tdm", "valu",
         "mma"], "V": ["control", "scalar", "lds", "lds_store", "tdm", "mma"]
    }
    payload_reads = waves * (wave_m * 128 + wave_n * 128 * weight_bits // 8) * substeps
    scale_reads = waves * (wave_m + wave_n) * 4 * substeps
    return dict(
        name=f"A8W{weight_bits} {tile_m}x{tile_n} BK{block_k}, {waves_m}x{waves_n} waves, {memory_path} path",
        clock="one shader-clock cycle", parameters=parameters, resources=resources, operations=operations,
        dependencies=dependencies, buffers=buffers, queues=queues, storage=storage, requests=requests, waits=waits,
        residency=residency, code=dict(
            total_bytes=None, hot_loop_bytes=None, tail_bytes=None, interpretation=
            "static footprint is evidence, not dynamic work; represent fetch delays as explicit operations"),
        coexecution={"mma_execute": [classes[c] for c in "0EEIEEISVV"]},
        compute_resources=[f"xdl{s}" for s in range(4)], workload=dict(
            tile_m=tile_m, tile_n=tile_n, k_step=block_k, native_k=128, waves=waves, waves_m=waves_m, waves_n=waves_n,
            weight_bits=weight_bits, packet_bytes=packet_bytes, mma_order=mma_order, array_mapping=array_mapping,
            wave_partitions=wave_partitions, lds_release=lds_release, transport=transport, lds_order=lds_order,
            stage_queues=stage_queues, array_arbitration=array_arbitration, scale_read_bytes=scale_read_bytes,
            memory_path=memory_path, flops_per_period=2 * tile_m * tile_n * block_k,
            wmma_instructions=tile_m * tile_n // 256 * substeps, unique_input_bytes=sum(sizes.values()),
            lds_payload_read_bytes=payload_reads, lds_scale_read_bytes=scale_reads,
            lds_instructions=sum(len(entries) for entries in lds_entries.values()),
            array_bytes_per_period=payload_reads + scale_reads + sum(sizes.values()), fp32_output_bytes=output_bytes,
            staged_output_array_bytes=2 * output_bytes, scalar_instructions=waves * scalar_ops,
            vector_instructions=waves * vector_ops * substeps, control_instructions=waves * substeps,
            vgprs_per_wave_expression=vgprs),
        assumptions=[
            "A design-space graph with a chosen matrix order, not a model of an existing kernel's instruction order.",
            "One CTA on one CU; four SIMD matrix engines, round-robin wave placement, adjacent-SIMD DS pairing.",
            "Rectangular wave ownership, native K128 operands, packed block-32 scales, b128 payload instructions.",
            "Physical BK transfers share visibility across their K128 halves; payload and scale rings are independent.",
            "S/A/B commands model partial fusion. Scales consume two descriptor credits; wave zero issues all commands.",
            "Input sector admission interleaves S/A/B. Array mapping is explicitly pooled or fixed-striped.",
            "Pooled array service allows either of two ports, at most one per request; striped fixes DS and packet mapping.",
            "Each array request holds a whole port for its rounded service duration; fractional bytes do not create extra ports.",
            ("LDS loads and TDM inputs have separately scheduled stages; contention extends request lifetimes."
             if transport == "staged" else
             "Fixed-route reservations delay issue for future conflicts; they do not reproduce internal queueing."),
            "Output uses fixed-route reservations. Input and output are serial in this single-tile graph.",
            "Two array ports serve both TDM and DS traffic; two distinct LDS return links serve only register loads.",
            ("DS scheduler entries remain occupied until array admission; a finite TDM return FIFO holds packets at ingress."
             if stage_queues else
             "Stage waiting rooms are unbounded; DS scheduler entries cover only their fixed configured stage time."),
            "Stage-queue capacities, request units, grouping, admission order and release endpoints require calibration.",
            "TDM FIFO residence consumes an explicit part of tcp_output_cycles; the rest precedes FIFO admission.",
            ("Array arbitration counts nonpreemptive stage-request grants, not bytes; instruction/clause grouping is not inferred."
             if array_arbitration != "scheduler" else
             "Shared array arbitration follows the list scheduler's priorities; no hardware arbitration policy is inferred."
             ),
            "A distinct per-wave LDS instruction pool is held through result completion. Counter bit width is not its calibration.",
            "LDS admission order is explicit. Operand order groups A then B; interleaved order alternates matching fragments.",
            "Same-wave LDS results retire in admission order; internal stages may finish earlier. Mixed event types need separate contracts.",
            "Scale instruction width is independent of packed-scale bytes; use the final instruction stream to choose it.",
            "Interface widths, pairing, route composition and queue interpretations require separate calibration.",
            "Cache path bypasses memory-channel traffic; memory path misses every input. No inter-CTA reuse is assumed.",
            "Memory bandwidth is a configurable fair share across active CUs; cache links are independent by direction.",
            "Per-fragment register reuse waits for all final source reads. Packed scales have a separate small ring.",
            "LDS release is explicit: source read, per-operand register completion, or all-stage workgroup completion.",
            "Workgroup release records an earliest arrival after all stage issues. ISA ordering can delay that arrival.",
            "Workgroup release joins this stage's reads; add younger completions explicitly when an emitted wait also covers them.",
            "Register credits are acquired before loads issue, a conservative overwrite rule; no bank conflicts modeled.",
            "Scaled issue precedes matrix execution; LLVM 0EEIEEISVV gates issue at actual execution, not queue admission.",
            "Eight cycles of matrix service remain eight even when the next scaled issue fits at execution offset seven.",
            "Address work is explicit and adjustable; invariant setup is hoisted, instruction-cache misses are excluded.",
            "Address sets remain live through DS decode. Loop control follows the last matrix issue in each K128 step.",
            "Local visibility/reuse delays are supplied independently. There are no cluster barriers or multicast here.",
            "FP32 stores retain the WMMA-to-memory hazard. TDM output reads the array but not the LDS register-return link.",
            "Output drains after all loop work. Input/output LDS allocations reuse space; VGPR allocation rounds per wave.",
            "One-CTA launch limits do not establish aggregate residency capacity or hardware admission policy.",
            "Used VGPR counts do not establish allocation; calibrate vgpr_next_free from the descriptor before predicting residency.",
            "Predictions are conditional on the profile. Configured component values are not measured end-to-end latencies."
        ])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in (("tile-m", 256), ("tile-n", 256), ("waves-m", 2), ("waves-n", 2), ("data-slots", 2),
                          ("scale-slots", 3), ("register-slots", 1), ("scale-register-slots", 2), ("packet-bytes", 256),
                          ("scalar-ops", 8), ("vector-ops", 0)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--weight-bits", type=int, choices=(4, 8), default=8)
    p.add_argument("--block-k", type=int, choices=(128, 256), default=256)
    p.add_argument("--a-register-slots", type=int)
    p.add_argument("--b-register-slots", type=int)
    p.add_argument("--mma-order", choices=("mn", "nm"), default="mn")
    p.add_argument("--array-mapping", choices=("pooled", "striped"), default="pooled")
    p.add_argument("--lds-release", choices=("source", "completion", "workgroup"), default="workgroup",
                   help="LDS reuse fence: source-read bound, operand completion, or full-stage workgroup completion")
    p.add_argument("--transport", choices=("staged", "fixed"), default="staged",
                   help="queue at individual input stages, or reserve each complete route at fixed offsets")
    p.add_argument("--stage-queues", action="store_true",
                   help="hold DS scheduler entries through array admission and bound the TDM return FIFO")
    p.add_argument("--array-arbitration", choices=("scheduler", "round-robin", "tdm-priority", "lds-priority"),
                   default="scheduler", help="shared array request arbitration; requires staged transport")
    p.add_argument("--scale-read-bytes", type=int, choices=(128, 256, 512), default=256,
                   help="packed-scale bytes returned by one wave instruction; match the compiled load width")
    p.add_argument("--lds-order", choices=("operand", "interleaved"), default="interleaved")
    p.add_argument("--wave-partitions", type=lambda value: [int(v) for v in value.split(",")],
                   help="array port per wave, e.g. 0,1,0,1; requires striped array mapping")
    p.add_argument("--memory-path", choices=("cache", "memory"), default="cache")
    p.add_argument("-o", "--output", type=Path, required=True)
    args = vars(p.parse_args())
    output = args.pop("output")
    try:
        data = make_model(**args)
    except ValueError as exc:
        p.error(str(exc))
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
