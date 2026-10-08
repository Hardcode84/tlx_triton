"""Periodic resource/dependency model. No GPU runtime or compiler is imported.

The lower bound is optimistic. The list schedule is a constructive upper bound
for the supplied abstract operations, not a proof of an optimal ISA schedule.
All time values are integer ticks in one explicitly chosen clock domain.
"""

import argparse
import ast
from bisect import bisect_right
import copy
import hashlib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from itertools import product
from pathlib import Path


class Unknown(ValueError):
    pass


def number(expr, parameters, stack=()):
    """Evaluate arithmetic only; configuration files cannot execute Python."""
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        value = float(expr)
    elif isinstance(expr, str):

        def visit(node):
            if isinstance(node, ast.Constant):
                return number(node.value, parameters)
            if isinstance(node, ast.Name):
                if node.id not in parameters or parameters[node.id] is None:
                    raise Unknown(node.id)
                if node.id in stack:
                    raise ValueError(f"cyclic parameter expression: {' -> '.join(stack + (node.id,))}")
                return number(parameters[node.id], parameters, stack + (node.id, ))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = visit(node.operand)
                return -value if isinstance(node.op, ast.USub) else value
            if isinstance(node, ast.BinOp):
                a, b = visit(node.left), visit(node.right)
                operators = {
                    ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b, ast.Div: lambda: a / b,
                    ast.FloorDiv: lambda: a // b
                }
                if type(node.op) in operators:
                    return operators[type(node.op)]()
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
                functions = {"ceil": math.ceil, "max": max, "min": min}
                if node.func.id in functions:
                    return functions[node.func.id](*[visit(arg) for arg in node.args])
            raise ValueError(f"unsupported expression: {expr!r}")

        value = float(visit(ast.parse(expr, mode="eval").body))
    else:
        raise ValueError(f"expected a number or arithmetic expression, got {expr!r}")
    if not math.isfinite(value):
        raise ValueError(f"non-finite value: {expr!r}")
    return value


def ticks(value):
    if value < 0:
        raise ValueError("time cannot be negative")
    return math.ceil(value)


def window_metrics(start, end, service, stalls=None):
    """Summarize half-open intervals on explicitly named matrix resources.

    Partner waves on the same SIMD belong in the same service list. Stall
    categories may overlap; only their union can be subtracted from idle.
    An instruction's issue timestamp alone is not a stall interval.
    """
    if not math.isfinite(start) or not math.isfinite(end) or end <= start or not service:
        raise ValueError("a window needs positive duration and explicit compute resources")
    stalls = stalls or {}
    if set(stalls) - set(service):
        raise ValueError("stall scope contains an unreported compute resource")

    def merge(intervals):
        clipped = []
        for a, b in intervals:
            if not math.isfinite(a) or not math.isfinite(b) or b < a:
                raise ValueError("reversed interval")
            a, b = max(start, a), min(end, b)
            if a < b:
                clipped.append((a, b))
        result = []
        for a, b in sorted(clipped):
            if result and a <= result[-1][1]:
                result[-1] = (result[-1][0], max(b, result[-1][1]))
            else:
                result.append((a, b))
        return result

    def intersect(a, b):
        i = j = 0
        result = []
        while i < len(a) and j < len(b):
            lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
            if lo < hi:
                result.append((lo, hi))
            if a[i][1] <= b[j][1]:
                i += 1
            else:
                j += 1
        return result

    def length(intervals):
        return sum(b - a for a, b in intervals)

    rows = {}
    for resource, spans in service.items():
        busy = merge(spans)
        idle, cursor = [], start
        for a, b in busy:
            if cursor < a:
                idle.append((cursor, a))
            cursor = b
        if cursor < end:
            idle.append((cursor, end))
        exposed = {kind: intersect(merge(intervals), idle) for kind, intervals in stalls.get(resource, {}).items()}
        union = merge([span for intervals in exposed.values() for span in intervals])
        rows[resource] = dict(service_ticks=length(busy), idle_ticks=length(idle),
                              exposed_by_category={kind: length(intervals)
                                                   for kind, intervals in exposed.items()},
                              exposed_union_ticks=length(union), unattributed_idle_ticks=length(idle) - length(union))
    numerator = sum(row["service_ticks"] for row in rows.values())
    denominator = len(service) * (end - start)
    return dict(window=[start, end], resources=rows, service_ticks=numerator, available_resource_ticks=denominator,
                service_fraction=numerator / denominator, stall_categories_are_non_additive=True)


def latency_statistics(bins):
    """Combine count/sum latency bins without averaging per-bin means.

    The caller must supply bins with the same scope, unit, and complete time
    coverage. No samples means unknown latency, including min/max sentinels.
    """
    count = total = 0
    minima, maxima = [], []
    complete_minima = complete_maxima = True
    for sample in bins:
        n, value = sample["count"], sample["sum"]
        if not math.isfinite(n) or not math.isfinite(value) or n < 0 or int(n) != n or value < 0 or (not n and value):
            raise ValueError("invalid latency counter bin")
        if n:
            count += n
            total += value
            if sample.get("minimum") is not None:
                if not 0 <= sample["minimum"] <= value / n:
                    raise ValueError("latency minimum exceeds mean or is negative")
                minima.append(sample["minimum"])
            else:
                complete_minima = False
            if sample.get("maximum") is not None:
                if not math.isfinite(sample["maximum"]) or sample["maximum"] < value / n:
                    raise ValueError("latency maximum is below mean or non-finite")
                maxima.append(sample["maximum"])
            else:
                complete_maxima = False
    return dict(count=count, sum=total, mean=total / count if count else None,
                minimum=min(minima) if minima and complete_minima else None,
                maximum=max(maxima) if maxima and complete_maxima else None)


@dataclass(frozen=True)
class Use:
    resource: str
    offset: int
    duration: int
    rate: float
    work: float


@dataclass(frozen=True)
class Op:
    name: str
    phase: str
    latency: int
    uses: tuple
    kind: str
    domain: str | None
    window_domain: str | None = None


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    distance: int
    delay: int
    reason: str = "dependency"


class Arbiter:
    """Work-conserving arbitration among eligible class heads.

    A weight counts grants, not bytes or occupied cycles. Empty or blocked
    classes surrender their remaining quantum. FIFO order within a class is
    supplied by the caller from dependency-ready dates, not criticality.
    """

    def __init__(self, policy, classes):
        self.policy = policy
        self.classes = classes
        self.cursor = 0
        self.remaining = classes[0][1]

    def choose(self, heads):
        start = self.cursor if self.policy == "round-robin" else 0
        for offset in range(len(self.classes)):
            index = (start + offset) % len(self.classes)
            kind, weight = self.classes[index]
            if kind in heads:
                remaining = self.remaining if index == self.cursor else weight
                return heads[kind], index, remaining
        return None

    def grant(self, choice):
        _, index, remaining = choice
        if self.policy == "round-robin":
            self.cursor, self.remaining = index, remaining - 1
            if not self.remaining:
                self.cursor = (index + 1) % len(self.classes)
                self.remaining = self.classes[self.cursor][1]


class Model:

    def __init__(self, data, bindings=None, overrides=None):
        self.data = copy.deepcopy(data)
        self.parameter_metadata = copy.deepcopy(data.get("parameters", {}))
        profile = bindings or {}
        structured = "parameters" in profile
        self.calibration = copy.deepcopy(profile.get("metadata", {})) if structured else {}
        # Observations describe a measurement, not an additive route delay.
        self.observations = copy.deepcopy(profile.get("observations", [])) if structured else []
        if structured and "coexecution" in profile:
            self.data["coexecution"] = copy.deepcopy(profile["coexecution"])
        if structured and "code" in profile:
            self.data["code"] = copy.deepcopy(profile["code"])
        for entries, source in ((profile.get("parameters", {}) if structured else profile, "bindings"),
                                (overrides or {}, "command-line override")):
            unknown = set(entries) - self.parameter_metadata.keys()
            if unknown:
                raise ValueError(f"unknown parameter bindings: {sorted(unknown)}")
            for name, entry in entries.items():
                update = entry if isinstance(entry, dict) else dict(value=entry, source=source, kind="override")
                if "value" not in update:
                    raise ValueError(f"parameter binding has no value: {name}")
                self.parameter_metadata[name].update(update)
        self.parameters = {k: v["value"] for k, v in self.parameter_metadata.items()}
        self.resolved_parameters = {}
        for name, value in self.parameters.items():
            try:
                self.resolved_parameters[name] = number(name, self.parameters) if value is not None else None
            except Unknown:
                self.resolved_parameters[name] = None
        self._value_cache = {}
        self.raw_ops = {op["id"]: op for op in self.data["operations"]}
        if len(self.raw_ops) != len(data["operations"]):
            raise ValueError("duplicate operation id")
        self.resources = self.data["resources"]
        windows = self.data.get("coexecution", {})
        if not isinstance(windows, dict) or any(
                not isinstance(pattern, list) or any(not isinstance(slot, list) or any(not isinstance(kind, str)
                                                                                       for kind in slot)
                                                     for slot in pattern)
                for pattern in windows.values()):
            raise ValueError("coexecution rules must map classes to lists of allowed-class lists")
        self.raw_edges = list(self.data.get("dependencies", []))
        self.waits = self.data.get("waits", [])
        self.requests = self.data.get("requests", [])
        self.add_wait_edges()
        # Ordered retirement normalizes release points without changing the
        # input graph or the readiness of its internal transport stages.
        self.queues = copy.deepcopy(self.data.get("queues", []))
        self.unresolved_queues = {}
        self.add_queue_edges()
        self.buffers = data.get("buffers", [])
        for buf in self.buffers:
            slots = self.value(buf["slots"])
            if slots < 1 or int(slots) != slots or not buf["consumers"]:
                raise ValueError(f"invalid ring depth or empty consumers: {buf['id']}")
            for consumer in buf["consumers"]:
                edge = dict(source=consumer, target=buf["producer"], distance=int(slots), reason=f"reuse:{buf['id']}")
                if "release_delay" in buf:
                    edge["delay"] = buf["release_delay"]
                self.raw_edges.append(edge)
        for op in self.raw_ops.values():
            if op.get("phase", "loop") not in ("prologue", "loop", "epilogue"):
                raise ValueError(f"invalid phase for {op['id']}")
            for use in op.get("uses", []):
                if use["resource"] not in self.resources:
                    raise ValueError(f"unknown resource: {use['resource']}")
            if op.get("domain") is not None and op["domain"] not in self.resources:
                raise ValueError("an issue domain must name a resource")
            if op.get("domain") is not None and not any(u["resource"] == op["domain"] for u in op.get("uses", [])):
                raise ValueError("an operation must reserve its declared issue domain")
            if op.get("window_domain") is not None and op["window_domain"] not in self.resources:
                raise ValueError("a window domain must name a resource")
        for edge in self.raw_edges:
            if edge["source"] not in self.raw_ops or edge["target"] not in self.raw_ops:
                raise ValueError(f"unknown dependency endpoint: {edge}")
            distance = edge.get("distance", 0)
            if type(distance) is not int or distance < 0:
                raise ValueError("dependency distance must be a nonnegative integer")
            src_phase = self.raw_ops[edge["source"]].get("phase", "loop")
            dst_phase = self.raw_ops[edge["target"]].get("phase", "loop")
            phases = {"prologue": 0, "loop": 1, "epilogue": 2}
            if phases[src_phase] > phases[dst_phase] or (distance and (src_phase != "loop" or dst_phase != "loop")):
                raise ValueError(f"invalid dependency between phases: {edge}")
        self.order = self.topological_order()
        self.validate_requests()
        self.arbitration = {}
        self.arbitrated_ops = {}
        for resource, spec in self.resources.items():
            if "arbitration" not in spec:
                continue
            arb = spec["arbitration"]
            classes = arb.get("classes", [])
            if arb.get("policy") not in ("round-robin", "priority") or not classes:
                raise ValueError(f"invalid arbitration policy/classes for {resource}")
            kinds = [entry["kind"] for entry in classes]
            if len(set(kinds)) != len(kinds) or any(not isinstance(kind, str) for kind in kinds):
                raise ValueError(f"arbitration classes must have unique names: {resource}")
            for entry in classes:
                try:
                    weight = self.value(entry.get("weight", 1))
                except Unknown:
                    continue
                if weight < 1 or int(weight) != weight:
                    raise ValueError("arbitration weights must be positive integer grant counts")
            members = []
            for name, op in self.raw_ops.items():
                uses = [u for u in op.get("uses", []) if u["resource"] == resource]
                if not uses or op.get("kind", "other") not in kinds:
                    continue
                if name in self.arbitrated_ops or len(uses) != 1 or self.value(uses[0].get("offset", 0)) != 0:
                    raise ValueError("arbitrated operations need one arbitration resource at offset zero")
                self.arbitrated_ops[name] = resource
                members.append(name)
            if set(kinds) - {self.raw_ops[name].get("kind", "other") for name in members}:
                raise ValueError(f"arbitration class has no operations on {resource}")
            self.arbitration[resource] = copy.deepcopy(arb)
        for resource in data.get("compute_resources", []):
            if resource not in self.resources:
                raise ValueError(f"unknown compute resource: {resource}")

    def value(self, expr):
        if expr not in self._value_cache:
            self._value_cache[expr] = number(expr, self.resolved_parameters)
        return self._value_cache[expr]

    @staticmethod
    def endpoint(point):
        return dict(op=point) if isinstance(point, str) else dict(point)

    def add_wait_edges(self):
        """A wait has an issue-frontier arrival and an explicit completion set.

        Include younger requests when an emitted counter wait covers them.
        The gate cannot open until every supplied completion has arrived.
        A compiler scheduling marker alone is not a completion dependency.
        """
        names = set()
        for wait in self.waits:
            if wait["id"] in names:
                raise ValueError(f"duplicate wait: {wait['id']}")
            names.add(wait["id"])
            if set(wait.get("compute_resources", [])) - set(self.data.get("compute_resources", [])):
                raise ValueError("wait compute_resources must name declared compute resources")
            if not wait["completions"]:
                raise ValueError(f"empty wait completion set: {wait['id']}")
            for point, role in [(wait["arrival"], "arrival")] + [(p, "completion") for p in wait["completions"]]:
                point = self.endpoint(point)
                edge = dict(source=point["op"], target=wait["target"], distance=point.get("distance", 0),
                            reason=f"wait-{role}:{wait['id']}")
                if "delay" in point:
                    edge["delay"] = point["delay"]
                self.raw_edges.append(edge)

    def validate_requests(self):
        names = set()
        forward = defaultdict(list)
        for edge in self.raw_edges:
            if not edge.get("distance", 0):
                forward[edge["source"]].append(edge["target"])

        def follows(source, target):
            pending, seen = [source], set()
            while pending:
                name = pending.pop()
                if name == target:
                    return True
                if name not in seen:
                    seen.add(name)
                    pending.extend(forward[name])
            return False

        for request in self.requests:
            if request["id"] in names:
                raise ValueError(f"duplicate request: {request['id']}")
            names.add(request["id"])
            if type(request.get("aggregate", False)) is not bool:
                raise ValueError(f"request aggregate must be a boolean: {request['id']}")
            chain = [request["issue"], *request.get("stages", []), request["complete"]]
            if any(name not in self.raw_ops for name in chain):
                raise ValueError(f"unknown request endpoint: {request['id']}")
            if len({self.raw_ops[name].get("phase", "loop") for name in chain}) != 1:
                raise ValueError(f"request route crosses serial phases: {request['id']}")
            if any(not follows(a, b) for a, b in zip(chain, chain[1:])):
                raise ValueError(f"request stages must follow issue in route order: {request['id']}")
            for point in request.get("consumers",
                                     []) + ([request["source_release"]] if "source_release" in request else []):
                point = self.endpoint(point)
                if point["op"] not in self.raw_ops or type(point.get("distance", 0)) is not int or point.get(
                        "distance", 0) < 0:
                    raise ValueError(f"invalid request checkpoint: {request['id']}")
            for raw in request.get("consumers", []):
                point = self.endpoint(raw)
                if not point.get("distance", 0) and not follows(request["complete"], point["op"]):
                    raise ValueError(f"request consumer does not depend on completion: {request['id']}")
            if "source_release" in request:
                point = self.endpoint(request["source_release"])
                if point.get("distance", 0) or not follows(request["issue"], point["op"]) or not follows(
                        point["op"], request["complete"]):
                    raise ValueError(f"request source release must lie on its completion path: {request['id']}")
            if request.get("compute_resource") is not None and request["compute_resource"] not in self.data.get(
                    "compute_resources", []):
                raise ValueError("request compute_resource must name a compute resource")

    def add_queue_edges(self):
        """Allocate a circular pool of credits in an explicit acquisition order.

        Each acquisition can own several credits. A future acquisition waits
        for every release whose credits it reuses. This is a realizable ordered
        allocation policy, not an assertion about a particular device's arbiter.
        Ordered retirement holds a ready entry until all older entries retire.
        Zero-latency retirement events keep this distinct from raw completion,
        including when release operations have different completion delays.
        """
        names = set()
        forward = defaultdict(list)
        for edge in self.raw_edges:
            if edge.get("distance", 0) == 0:
                forward[edge["source"]].append(edge["target"])

        def reachable(source, target):
            pending, seen = [source], set()
            while pending:
                node = pending.pop()
                if node == target:
                    return True
                if node not in seen:
                    seen.add(node)
                    pending.extend(forward[node])
            return False

        for queue in self.queues:
            name = queue["id"]
            if name in names:
                raise ValueError(f"duplicate queue: {name}")
            names.add(name)
            try:
                capacity = self.value(queue["capacity"])
            except Unknown as exc:
                capacity = None
                self.unresolved_queues[name] = str(exc)
            if capacity is not None and (capacity < 1 or int(capacity) != capacity):
                raise ValueError(f"queue {name} needs a positive integer credit capacity")
            entries = queue["entries"]
            if not entries:
                raise ValueError(f"empty queue: {name}")
            ordered = queue.get("ordered_retirement", False)
            if type(ordered) is not bool:
                raise ValueError(f"queue {name} ordered_retirement must be a boolean")
            ends, total, phases = [], 0, set()
            for entry in entries:
                if "retire" in entry and not ordered:
                    raise ValueError(f"queue {name} needs ordered_retirement to name a retirement event")
                for endpoint in ("acquire", "release"):
                    if entry[endpoint] not in self.raw_ops:
                        raise ValueError(f"unknown queue endpoint: {entry[endpoint]}")
                    phases.add(self.raw_ops[entry[endpoint]].get("phase", "loop"))
                if not reachable(entry["acquire"], entry["release"]):
                    raise ValueError(f"queue {name} releases credits without depending on their acquisition")
                units = self.value(entry.get("units", 1))
                if units < 1 or int(units) != units or (capacity is not None and units > capacity):
                    raise ValueError(f"invalid credit demand for queue {name}: {units}")
                total += int(units)
                ends.append(total)
            if len(phases) != 1:
                raise ValueError(f"queue {name} crosses serial phases")
            cyclic = phases == {"loop"}
            added = set()

            def edge(source, target, distance, delay, reason):
                key = (source, target, distance, str(delay), reason)
                if key in added:
                    return
                added.add(key)
                item = dict(source=source, target=target, distance=distance, reason=reason)
                if delay is not None:
                    item["delay"] = delay
                self.raw_edges.append(item)
                if distance == 0:
                    forward[source].append(target)

            if ordered:
                for index, entry in enumerate(entries):
                    retired = entry.get("retire", f"queue_retire:{name}:{index}")
                    if "retire" in entry:
                        raw = self.raw_ops.get(retired)
                        if (raw is None or raw.get("phase", "loop") not in phases or raw.get("uses")
                                or self.value(raw.get("latency", 0)) != 0):
                            raise ValueError(f"queue {name} needs a zero-latency retirement event in the same phase")
                    else:
                        if retired in self.raw_ops:
                            raise ValueError(f"generated queue retirement id already exists: {retired}")
                        self.raw_ops[retired] = dict(id=retired, phase=next(iter(phases)), kind="retirement", latency=0)
                    edge(entry["release"], retired, 0, entry.get("release_delay"), f"queue-ready:{name}")
                    entry["release"] = retired
                    entry.pop("release_delay", None)

            for index, entry in enumerate(entries):
                if index + 1 < len(entries):
                    edge(entry["acquire"], entries[index + 1]["acquire"], 0, 0, f"queue-order:{name}")
                    if ordered:
                        edge(entry["release"], entries[index + 1]["release"], 0, 0, f"queue-retire:{name}")
                elif cyclic:
                    edge(entry["acquire"], entries[0]["acquire"], 1, 0, f"queue-order:{name}")
                    if ordered:
                        edge(entry["release"], entries[0]["release"], 1, 0, f"queue-retire:{name}")
                if capacity is None:
                    # Preserve admission order for a partial lower bound. A
                    # schedule must never silently omit the capacity edges.
                    continue
                begin = ends[index - 1] if index else 0
                for credit in range(begin, ends[index]):
                    distance, position = divmod(credit + int(capacity), total)
                    if distance and not cyclic:
                        continue
                    target = entries[bisect_right(ends, position)]["acquire"]
                    edge(entry["release"], target, distance, entry.get("release_delay"), f"queue-reuse:{name}")

    def topological_order(self):
        indegree = dict.fromkeys(self.raw_ops, 0)
        successors = defaultdict(list)
        for edge in self.raw_edges:
            if edge.get("distance", 0) == 0:
                indegree[edge["target"]] += 1
                successors[edge["source"]].append(edge["target"])
        ready = deque(k for k, n in indegree.items() if n == 0)
        order = []
        while ready:
            src = ready.popleft()
            order.append(src)
            for dst in successors[src]:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    ready.append(dst)
        if len(order) != len(self.raw_ops):
            raise ValueError("zero-distance dependency graph must be acyclic")
        return order

    def resolve(self, phases=None):
        selected = {
            name: op
            for name, op in self.raw_ops.items()
            if phases is None or op.get("phase", "loop") in phases
        }
        required = {u["resource"] for op in selected.values() for u in op.get("uses", [])}
        required.update(self.data.get("compute_resources", []))
        capacities = {k: self.value(v["capacity"]) for k, v in self.resources.items() if k in required}
        if any(v <= 0 for v in capacities.values()):
            raise ValueError("resource capacities must be positive")
        ops = {}
        for name, raw in selected.items():
            uses = []
            for use in raw.get("uses", []):
                resource = use["resource"]
                work = self.value(use["work"])
                rate = self.value(use["rate"]) if "rate" in use else capacities[resource]
                if work < 0 or not 0 < rate <= capacities[resource]:
                    raise ValueError(f"invalid work/rate in {name}: {use}")
                duration = ticks(work / rate)
                # Rounding extends the reservation; it does not create extra work.
                uses.append(
                    Use(resource, ticks(self.value(use.get("offset", 0))), duration, work / duration if duration else 0,
                        work))
            latency = max([ticks(self.value(raw.get("latency", 0)))] + [u.offset + u.duration for u in uses])
            demand = defaultdict(lambda: defaultdict(float))
            for use in uses:
                demand[use.resource][use.offset] += use.rate
                demand[use.resource][use.offset + use.duration] -= use.rate
            for resource, changes in demand.items():
                occupied = 0
                for _, change in sorted(changes.items()):
                    occupied += change
                    if occupied > capacities[resource] + 1e-8:
                        raise ValueError(f"operation {name} alone exceeds capacity of {resource}")
            ops[name] = Op(name, raw.get("phase", "loop"), latency, tuple(uses), raw.get("kind", "other"),
                           raw.get("domain"), raw.get("window_domain", raw.get("domain")))
        edges = [
            Edge(e["source"], e["target"], e.get("distance", 0),
                 ticks(self.value(e["delay"])) if "delay" in e else ops[e["source"]].latency,
                 e.get("reason", "dependency")) for e in self.raw_edges if e["source"] in ops and e["target"] in ops
        ]
        return capacities, ops, edges

    def storage(self):
        result = {}
        for space, spec in self.data.get("storage", {}).items():
            allocations = [(b["id"], self.value(b["slots"]) * self.value(b["size"]))
                           for b in self.buffers
                           if b.get("space") == space]
            fixed = self.value(spec.get("fixed", 0))
            by_phase = {
                phase: fixed + self.value(spec.get("phase_fixed", {}).get(phase, 0))
                for phase in ("prologue", "loop", "epilogue")
            }
            by_phase["loop"] += sum(size for _, size in allocations)
            required = max(by_phase.values())
            if fixed < 0 or min(by_phase.values()) < 0 or any(size < 0 for _, size in allocations):
                raise ValueError("storage sizes cannot be negative")
            capacity = None if spec.get("capacity") is None else self.value(spec["capacity"])
            allocated = self.value(spec["allocated"]) if "allocated" in spec else required
            if allocated < 0:
                raise ValueError("allocated storage cannot be negative")
            result[space] = dict(
                required=required, allocated=allocated, capacity=capacity, covers_modeled_requirement=allocated
                >= required, allocation_basis=spec.get(
                    "allocation_basis",
                    "declared allocation" if "allocated" in spec else "modeled storage rounded by the graph"),
                fits=False if allocated < required else None if capacity is None else allocated <= capacity,
                unit=spec.get("unit", "bytes"), fixed=fixed, by_phase=by_phase, buffers=dict(allocations))
        return result

    def residency(self):
        """Necessary admission limits, separate from per-workgroup legality."""
        rows = []
        for spec in self.data.get("residency", []):
            row = dict(resource=spec["resource"], scope=spec.get("scope", "unspecified"),
                       source=spec.get("source", "unspecified"))
            try:
                capacity = self.value(spec["capacity"]) if spec.get("capacity") is not None else None
                per_cta = self.value(spec["per_cta"])
                if per_cta <= 0 or (capacity is not None and capacity < 0):
                    raise ValueError("invalid residency capacity or per-CTA allocation")
                row.update(capacity=capacity, per_cta=per_cta,
                           cta_upper_bound=math.floor(capacity / per_cta) if capacity is not None else None)
            except Unknown as exc:
                row.update(cta_upper_bound=None, unknown=str(exc))
            rows.append(row)
        known = [row["cta_upper_bound"] for row in rows if row["cta_upper_bound"] is not None]
        return dict(
            constraints=rows, upper_bound_from_known_constraints=min(known) if known else None,
            all_constraints_resolved=bool(rows) and all(row["cta_upper_bound"] is not None for row in rows),
            interpretation="necessary capacity limits; admission policy and launch coverage require separate evidence")

    @staticmethod
    def recurrence(ops, edges):
        """Exact maximum cycle ratio, including fractional average periods.

        Positive-cycle relaxation uses integer-scaled weights. Each discovered
        cycle increases the bound to its exact ratio; no rounded or sampled
        latency is substituted for a recurrence constraint.
        """
        names = [k for k, op in ops.items() if op.phase == "loop"]
        name_set = set(names)
        cyclic = [e for e in edges if e.source in name_set and e.target in name_set]
        outgoing = defaultdict(list)
        for edge in cyclic:
            outgoing[edge.source].append(edge)

        def extract_cycle(predecessor, node):
            for _ in names:
                node = predecessor[node].source
            first, cycle = node, []
            while True:
                edge = predecessor[node]
                cycle.append(edge)
                node = edge.source
                if node == first:
                    return cycle

        def full_relaxation(period):
            starts = dict.fromkeys(names, 0)
            predecessor = {}
            for _ in names:
                changed = None
                for edge in cyclic:
                    candidate = starts[edge.source] + edge.delay * period.denominator - edge.distance * period.numerator
                    if starts[edge.target] < candidate:
                        starts[edge.target] = candidate
                        predecessor[edge.target] = edge
                        changed = edge.target
                if changed is None:
                    return None
            if not names:
                return None
            return extract_cycle(predecessor, changed)

        def positive_cycle(period):
            # Revisit only changed vertices. Packetized graphs have many
            # independent transfers; rescanning every edge N times for each
            # positive-cycle witness makes parameter sweeps needlessly costly.
            starts = dict.fromkeys(names, 0)
            depth = dict.fromkeys(names, 0)
            predecessor = {}
            pending, queued = deque(names), set(names)
            while pending:
                source = pending.popleft()
                queued.remove(source)
                for edge in outgoing[source]:
                    candidate = starts[source] + edge.delay * period.denominator - edge.distance * period.numerator
                    target = edge.target
                    if starts[target] >= candidate:
                        continue
                    starts[target] = candidate
                    predecessor[target] = edge
                    depth[target] = depth[source] + 1
                    # A wrap edge often closes a short critical cycle long
                    # before the N-edge path threshold. Certify that cycle
                    # directly instead of traversing it N times.
                    if edge.distance:
                        node, seen, path = target, {}, []
                        for _ in range(min(256, len(names))):
                            if node in seen:
                                cycle = path[seen[node]:]
                                weight = sum(e.delay * period.denominator - e.distance * period.numerator
                                             for e in cycle)
                                if weight > 0:
                                    return cycle
                                break
                            if node not in predecessor:
                                break
                            seen[node] = len(path)
                            path.append(predecessor[node])
                            node = predecessor[node].source
                    if depth[target] >= len(names):
                        try:
                            cycle = extract_cycle(predecessor, target)
                        except KeyError:
                            return full_relaxation(period)
                        weight = sum(e.delay * period.denominator - e.distance * period.numerator for e in cycle)
                        if weight <= 0:
                            return full_relaxation(period)
                        return cycle
                    if target not in queued:
                        pending.append(target)
                        queued.add(target)
            return None

        bound, witness = Fraction(0), []
        for _ in range(10000):
            cycle = positive_cycle(bound)
            if cycle is None:
                return float(bound), [
                    dict(source=e.source, target=e.target, delay=e.delay, distance=e.distance, reason=e.reason)
                    for e in witness
                ]
            candidate = Fraction(sum(e.delay for e in cycle), sum(e.distance for e in cycle))
            if candidate <= bound:
                raise AssertionError("positive cycle failed to increase the recurrence bound")
            bound, witness = candidate, cycle
        raise ValueError("too many critical cycles while finding recurrence bound")

    def bounds(self, target_period=None):
        rows = []
        unresolved = set()
        for resource, spec in self.resources.items():
            uses = [
                u for op in self.raw_ops.values() if op.get("phase", "loop") == "loop" for u in op.get("uses", [])
                if u["resource"] == resource
            ]
            fixed_uses = [
                u for op in self.raw_ops.values() if op.get("phase", "loop") != "loop" for u in op.get("uses", [])
                if u["resource"] == resource
            ]
            row = dict(resource=resource, unit=spec.get("unit", "work/tick"), capacity_expression=spec["capacity"])
            known_work = 0
            symbolic_work = Counter()
            for use in uses:
                try:
                    known_work += self.value(use["work"])
                except Unknown:
                    symbolic_work[str(use["work"])] += 1
            terms = ([f"{known_work:g}"] if known_work else []) + [
                expression if count == 1 else f"{count} * ({expression})"
                for expression, count in symbolic_work.items()
            ]
            row["work_expression"] = " + ".join(terms) or "0"
            try:
                row["work_per_period"] = sum(self.value(u["work"]) for u in uses)
                row["fixed_work"] = sum(self.value(u["work"]) for u in fixed_uses)
                cap = self.value(spec["capacity"])
                if cap <= 0:
                    raise ValueError("resource capacities must be positive")
                row.update(capacity=cap, cycles=row["work_per_period"] / cap)
                if target_period is not None:
                    row["load_at_target"] = row["cycles"] / target_period
            except Unknown as exc:
                unresolved.add(str(exc))
                row["unknown"] = str(exc)
            rows.append(row)
        result = dict(name=self.data.get("name", "unnamed"), resources=rows,
                      assumptions=self.data.get("assumptions", []), parameter_values=self.resolved_parameters,
                      parameter_metadata=self.parameter_metadata, calibration=self.calibration,
                      observations=self.observations, clock=self.data.get("clock", "unspecified ticks"),
                      workload=self.data.get("workload", {}), code=self.data.get("code", {}))
        result["arbitration"] = self.arbitration
        result["residency"] = self.residency()
        known_bound = max([0] + [r["cycles"] for r in rows if "cycles" in r])
        try:
            _, ops, edges = self.resolve({"loop"})
            rec, witness = self.recurrence(ops, edges)
            result["recurrence_cycles"] = rec
            result["recurrence_witness"] = witness
            known_bound = max(known_bound, rec)
            result["queue_lifetimes"] = self.queue_leads(ops, edges, target_period)
            known_bound = max(
                [known_bound] +
                [q["minimum_period"] for q in result["queue_lifetimes"] if q["minimum_period"] is not None])
            if target_period:
                result["buffers"] = self.buffer_leads(ops, edges, target_period)
        except Unknown as exc:
            unresolved.add(str(exc))
        try:
            result["storage"] = self.storage()
        except Unknown as exc:
            unresolved.add(str(exc))
        unresolved.update(k for k, v in self.resolved_parameters.items() if v is None)
        result["queues"] = [
            dict(id=q["id"], capacity=None if q["id"] in self.unresolved_queues else self.value(q["capacity"]),
                 entries=len(q["entries"]), phase=self.raw_ops[q["entries"][0]["acquire"]].get("phase", "loop"),
                 credits_per_phase=sum(self.value(e.get("units", 1))
                                       for e in q["entries"]), policy="ordered circular credits",
                 ordered_retirement=q.get("ordered_retirement", False), role=q.get("role", "unspecified"),
                 scope=q.get("scope", "unspecified"), unit=q.get("unit", "credits"))
            for q in self.queues
        ]
        result["unresolved_queues"] = dict(self.unresolved_queues)
        fits = [row["fits"] for row in result.get("storage", {}).values()]
        result["storage_feasible"] = (False
                                      if False in fits else None if "storage" not in result or None in fits else True)
        result["unresolved"] = sorted(unresolved)
        result["period_lower_bound"] = known_bound
        result["integer_period_lower_bound"] = ticks(known_bound)
        result["bound_is_partial"] = bool(unresolved)
        compute = set(self.data.get("compute_resources", []))
        compute_rows = [r for r in rows if r["resource"] in compute]
        if compute_rows and all("cycles" in r for r in compute_rows) and known_bound:
            compute_cycles = sum(r["work_per_period"] for r in compute_rows) / sum(r["capacity"] for r in compute_rows)
            result["compute_service_cycles"] = compute_cycles
            result["compute_utilization_upper_bound"] = min(1, compute_cycles / known_bound)
        return result

    def queue_leads(self, ops, edges, target_period=None):
        """Dependency-only credit lifetimes and the corresponding area bound.

        In a repeating schedule, capacity * period must cover the sum of
        credits * lifetime. Queueing or resource conflicts can extend lifetimes;
        these minima are necessary conditions, not sufficient queue depths.
        """
        forward = defaultdict(list)
        for edge in edges:
            if edge.distance == 0:
                forward[edge.source].append(edge)
        order_index = {name: i for i, name in enumerate(self.order)}
        rows = []
        for queue in self.queues:
            if any(e["acquire"] not in ops or ops[e["acquire"]].phase != "loop" for e in queue["entries"]):
                continue
            lifetimes, area = [], 0
            for entry in queue["entries"]:
                source, target = entry["acquire"], entry["release"]
                distance = {source: 0}
                if source != target:
                    for name in self.order[order_index[source]:order_index[target]]:
                        if name in distance:
                            for edge in forward[name]:
                                distance[edge.target] = max(distance.get(edge.target, 0), distance[name] + edge.delay)
                if target not in distance:
                    raise ValueError(f"queue {queue['id']} release does not follow acquisition")
                lifetime = distance[target] + (ticks(self.value(entry["release_delay"]))
                                               if "release_delay" in entry else ops[target].latency)
                lifetimes.append(lifetime)
                area += self.value(entry.get("units", 1)) * lifetime
            capacity = None if queue["id"] in self.unresolved_queues else self.value(queue["capacity"])
            row = dict(queue=queue["id"], capacity=capacity, minimum_lifetime=min(lifetimes),
                       maximum_minimum_lifetime=max(lifetimes), minimum_credit_ticks_per_period=area,
                       minimum_period=area / capacity if capacity is not None else None)
            if target_period is not None:
                row["minimum_credits_at_target"] = math.ceil(area / target_period)
            rows.append(row)
        return rows

    def buffer_leads(self, ops, edges, period):
        rows = []
        forward = defaultdict(list)
        for edge in edges:
            if edge.distance == 0:
                forward[edge.source].append(edge)
        for buf in self.buffers:
            distance = {buf["producer"]: 0}
            for name in self.order:
                if name in distance:
                    for edge in forward[name]:
                        distance[edge.target] = max(distance.get(edge.target, 0), distance[name] + edge.delay)
            if any(c not in distance for c in buf["consumers"]):
                raise ValueError(f"buffer {buf['id']} has a consumer not dependent on its producer")
            lifetime = max(distance[c] +
                           (ticks(self.value(buf["release_delay"])) if "release_delay" in buf else ops[c].latency)
                           for c in buf["consumers"])
            slots = int(self.value(buf["slots"]))
            rows.append(
                dict(buffer=buf["id"], minimum_lifetime=lifetime, slots=slots,
                     minimum_slots_at_target=math.ceil(lifetime / period),
                     reuse_slack_at_target=slots * period - lifetime))
        return rows

    def check_periodic_template(self, starts, length, period, capacities, ops, edges):
        """Check a complete cyclic template, including resource/window wrap."""
        if self.arbitration:
            return None
        if type(length) is not int or length < 1 or type(period) is not int or period < 1:
            raise ValueError("a periodic template needs a positive integer length and period")
        expected = {(name, j) for name, op in ops.items() if op.phase == "loop" for j in range(length)}
        if set(starts) != expected:
            raise ValueError("a periodic template must contain every loop operation exactly once per iteration")
        names = {name for name, _ in starts}
        windows = self.data.get("coexecution", {})
        for edge in edges:
            if edge.source not in names or edge.target not in names:
                continue
            for j in range(length):
                quotient, remainder = divmod(j - edge.distance, length)
                if starts[edge.target, j] < starts[edge.source, remainder] + quotient * period + edge.delay:
                    return None
        calendar = {r: defaultdict(float) for r in capacities}
        issues = defaultdict(lambda: defaultdict(list))
        for key, start in starts.items():
            op = ops[key[0]]
            for use in op.uses:
                for t in range(start + use.offset, start + use.offset + use.duration):
                    calendar[use.resource][t % period] += use.rate
                    if calendar[use.resource][t % period] > capacities[use.resource] + 1e-8:
                        return None
                    if use.resource == op.domain:
                        issues[op.domain][t % period].append((key, op.kind))
        for key, start in starts.items():
            op = ops[key[0]]
            for offset, allowed in enumerate(windows.get(op.kind, []) if op.window_domain is not None else []):
                for other, kind in issues[op.window_domain][(start + offset) % period]:
                    if (other != key or offset != 0) and kind not in allowed and "*" not in allowed:
                        return None
        utilization = {r: sum(cal.values()) / (capacities[r] * period) for r, cal in calendar.items()}
        compute = self.data.get("compute_resources", [])
        origin = min(starts.values())
        return dict(
            iterations=length, cycles=period, cycles_per_iteration=period / length, resource_utilization=utilization,
            compute_utilization=sum(utilization[r] * capacities[r]
                                    for r in compute) / sum(capacities[r] for r in compute) if compute else None,
            verified_infinite_wrap=True, template=[
                dict(op=name, iteration=j, start=start - origin)
                for (name, j), start in sorted(starts.items(), key=lambda item: item[1])
            ])

    def periodic_witness(self, placed, capacities, ops, edges, iterations, warmup):
        """Certify an observed repeat or an explicitly retimed cyclic template.

        Neither an interior average nor a list-scheduling fixed point is used
        as proof. Retimed templates preserve each operation's relative start
        within the sampled block, then search a bounded set of repeat periods.
        """
        # A feasible resource/dependency template does not certify a stateful
        # arbiter's grants across the wrap. Keep that result finite until the
        # arbiter state and work-conserving choices are checked periodically.
        if self.arbitration:
            return None
        names = [name for name, op in ops.items() if op.phase == "loop"]
        name_set = set(names)
        if not names:
            return None
        for length in range(1, max(1, (iterations - 2 * warmup) // 3) + 1):
            if warmup + length >= iterations:
                break
            period = placed[names[0], warmup + length] - placed[names[0], warmup]
            if period <= 0:
                continue
            if any(placed[name, i + length] - placed[name, i] != period
                   for name in names
                   for i in range(warmup, iterations - warmup - length)):
                continue
            starts = {(name, j): placed[name, warmup + j] for name in names for j in range(length)}
            witness = self.check_periodic_template(starts, length, period, capacities, ops, edges)
            if witness:
                witness.update(method="observed repeat", source_iterations=[warmup, warmup + length])
                return witness
        cyclic_edges = [e for e in edges if e.source in name_set and e.target in name_set]
        resource_bound = max([0] + [
            sum(u.work for name in names for u in ops[name].uses if u.resource == r) / c for r, c in capacities.items()
        ])
        best = None
        # Search a few interior blocks, not every possible modulo schedule.
        # Larger periods may work even if this bounded search finds no witness.
        for length in (1, 2):
            if iterations - 2 * warmup < length:
                continue
            first, last = warmup, iterations - warmup - length
            for base in sorted({first, (first + last) // 2, last}):
                starts = {(name, j): placed[name, base + j] for name in names for j in range(length)}
                minimum = math.ceil(resource_bound * length)
                for edge in cyclic_edges:
                    for j in range(length):
                        quotient, remainder = divmod(j - edge.distance, length)
                        if quotient < 0:
                            minimum = max(
                                minimum,
                                math.ceil(
                                    (starts[edge.source, remainder] + edge.delay - starts[edge.target, j]) / -quotient))
                limit = minimum + 256
                if best:
                    limit = min(limit, math.ceil(best["cycles_per_iteration"] * length))
                for period in range(max(1, minimum), limit):
                    witness = self.check_periodic_template(starts, length, period, capacities, ops, edges)
                    if witness:
                        witness.update(method="retimed template", source_iterations=[base, base + length])
                        best = witness
                        break
        return best

    def schedule(self, iterations=16, warmup=4, max_cycles=1000000, policy="critical-path", target_period=None,
                 analysis_windows=None):
        if self.unresolved_queues:
            raise Unknown("unfilled queue capacities: " + ", ".join(self.unresolved_queues.values()))
        capacities, ops, edges = self.resolve()
        original_ops = dict(ops)
        if iterations < 1 or warmup < 0 or 2 * warmup >= iterations:
            raise ValueError("require iterations > 2 * warmup >= 0")
        storage = self.storage()
        if any(row["fits"] is False for row in storage.values()):
            raise ValueError("declared storage allocation is insufficient or exceeds capacity")
        keys = [(name, i) for name, op in ops.items() for i in (range(iterations) if op.phase == "loop" else [-1])]
        indegree = dict.fromkeys(keys, 0)
        successors = defaultdict(list)
        expanded_edges = []
        for edge in edges:
            source, target = ops[edge.source], ops[edge.target]
            if source.phase == target.phase == "loop":
                pairs = [(i - edge.distance, i) for i in range(edge.distance, iterations)]
            elif source.phase == "loop":
                pairs = [(iterations - 1, -1)]
            elif target.phase == "loop":
                pairs = [(-1, 0)]
            else:
                pairs = [(-1, -1)]
            for src_i, dst_i in pairs:
                src, dst = (edge.source, src_i), (edge.target, dst_i)
                successors[src].append((dst, edge.delay))
                indegree[dst] += 1
                expanded_edges.append((src, dst, edge.delay, edge.reason))
        # One join per phase avoids a quadratic all-to-all dependency expansion
        # when packetized transfers have many independent nodes.
        phases = {phase: [k for k in keys if ops[k[0]].phase == phase] for phase in ("prologue", "loop", "epilogue")}
        incoming_same_phase = {dst for src, dst, _, _ in expanded_edges if ops[src[0]].phase == ops[dst[0]].phase}
        nonempty_phases = [phase for phase in phases if phases[phase]]
        for source_phase, target_phase in zip(nonempty_phases, nonempty_phases[1:]):
            roots = [k for k in phases[target_phase] if k not in incoming_same_phase]
            name = f"__end_{source_phase}"
            while name in ops:
                name += "_"
            gate = (name, -1)
            ops[name] = Op(name, source_phase, 0, (), "phase_boundary", None)
            keys.append(gate)
            indegree[gate] = 0
            for src in phases[source_phase]:
                delay = ops[src[0]].latency
                successors[src].append((gate, delay))
                indegree[gate] += 1
                expanded_edges.append((src, gate, delay, "phase boundary"))
            for dst in roots:
                successors[gate].append((dst, 0))
                indegree[dst] += 1
                expanded_edges.append((gate, dst, 0, "phase boundary"))
        # Longest remaining dependency chain prioritizes ready operations.
        priority = dict.fromkeys(keys, 0)
        remaining = dict(indegree)
        pending = deque(k for k, count in remaining.items() if count == 0)
        expanded_order = []
        while pending:
            key = pending.popleft()
            expanded_order.append(key)
            for dst, _ in successors[key]:
                remaining[dst] -= 1
                if remaining[dst] == 0:
                    pending.append(dst)
        if len(expanded_order) != len(keys):
            raise ValueError("expanded graph contains a cycle")
        for key in reversed(expanded_order):
            priority[key] = max([ops[key[0]].latency] + [delay + priority[dst] for dst, delay in successors[key]])
        finite_dependency_bound = max(priority.values(), default=0)
        if policy not in ("critical-path", "deadline"):
            raise ValueError("unknown scheduling policy")
        deadlines = dict.fromkeys(keys, math.inf)
        if policy == "deadline":
            compute_resources = set(self.data.get("compute_resources", []))
            service = {
                r:
                sum(u.work / capacities[r]
                    for op in original_ops.values()
                    if op.phase == "loop"
                    for u in op.uses if u.resource == r)
                for r in compute_resources
            }
            target_period = target_period or max(service.values(), default=1)
            if target_period <= 0:
                raise ValueError("target period must be positive")
            cursor = defaultdict(float)
            # Uniform ideal compute dates define urgency, not hard deadlines.
            # Propagating dates back to requests favors loads needed by the
            # next matrix instruction over unnecessary distant prefetches.
            for name in self.order:
                op = original_ops[name]
                uses = [u for u in op.uses if u.resource in compute_resources and op.phase == "loop"]
                if uses:
                    date = max(cursor[u.resource] * target_period / service[u.resource] - u.offset for u in uses)
                    for iteration in range(iterations):
                        deadlines[name, iteration] = iteration * target_period + date
                    for u in uses:
                        cursor[u.resource] += u.work / capacities[u.resource]
            for key in reversed(expanded_order):
                deadlines[key] = min([deadlines[key]] + [deadlines[dst] - delay for dst, delay in successors[key]])
        phase_work = {phase: defaultdict(float) for phase in phases}
        for name, _ in keys:
            for use in ops[name].uses:
                phase_work[ops[name].phase][use.resource] += use.work
        finite_resource_bound = sum(
            max([0] + [work / capacities[r] for r, work in amounts.items()]) for amounts in phase_work.values())
        ready = {k: 0 for k, n in indegree.items() if n == 0}
        earliest = dict.fromkeys(keys, 0)
        calendar = {k: defaultdict(float) for k in capacities}
        # Windows constrain issue classes on a shared issue domain. They are
        # supplied explicitly; an empty table assumes unrestricted coexecution.
        windows = self.data.get("coexecution", {})
        active = defaultdict(list)
        issued = defaultdict(lambda: defaultdict(list))
        placed = {}
        current = 0
        arbiters = {
            resource:
            Arbiter(spec["policy"], [(entry["kind"], int(self.value(entry.get("weight", 1))))
                                     for entry in spec["classes"]])
            for resource, spec in self.arbitration.items()
        }
        order_index = {name: index for index, name in enumerate(self.order)}
        grants = defaultdict(list)
        blocked = defaultdict(lambda: defaultdict(list))

        def record_block(key, reason):
            spans = blocked[key][reason]
            if spans and spans[-1][1] >= current:
                spans[-1][1] = max(spans[-1][1], current + 1)
            else:
                spans.append([current, current + 1])

        def issue_ticks(op, time):
            return [
                t for u in op.uses if u.resource == op.domain
                for t in range(time + u.offset, time + u.offset + u.duration)
            ]

        def blockers(op, time):
            reasons = set()
            for start, pattern in active[op.domain]:
                for t in issue_ticks(op, time):
                    offset = t - start
                    if 0 <= offset < len(pattern) and op.kind not in pattern[offset] and "*" not in pattern[offset]:
                        reasons.add(f"issue-window:{op.domain}")
            if op.kind in windows and op.window_domain is not None:
                pattern = windows[op.kind]
                if op.domain is not None and issue_ticks(op, time) != [time]:
                    raise ValueError("coexecution-window producers must have one issue tick at offset zero")
                for offset, allowed in enumerate(pattern):
                    if any(kind not in allowed and "*" not in allowed
                           for _, kind in issued[op.window_domain][time + offset]):
                        reasons.add(f"issue-window:{op.window_domain}")
            extra = defaultdict(float)
            for use in op.uses:
                for tick in range(time + use.offset, time + use.offset + use.duration):
                    extra[use.resource, tick] += use.rate
            reasons.update(f"resource:{r}" for (r, t), rate in extra.items()
                           if calendar[r][t] + rate > capacities[r] + 1e-8)
            return reasons

        def arbitrate(resource, candidates):
            heads = {}
            for candidate in candidates:
                if candidate not in ready or self.arbitrated_ops.get(candidate[0]) != resource:
                    continue
                kind = ops[candidate[0]].kind
                key = (ready[candidate], candidate[1], order_index[candidate[0]])
                if kind not in heads or key < heads[kind][0]:
                    heads[kind] = (key, candidate)
            eligible = {kind: key for kind, (_, key) in heads.items() if not blockers(ops[key[0]], current)}
            return arbiters[resource].choose(eligible)

        while ready:
            current = max(current, min(ready.values()))
            if current > max_cycles:
                raise ValueError("schedule exceeded max_cycles; check capacity, windows, and dependencies")
            candidates = sorted((k for k, time in ready.items() if time <= current), key=lambda k:
                                (deadlines[k] if policy == "deadline" else -priority[k], -priority[k], k[1], k[0]))
            progress = False
            for key in candidates:
                op = ops[key[0]]
                reasons = blockers(op, current)
                if reasons:
                    for reason in reasons:
                        record_block(key, reason)
                    continue
                resource = self.arbitrated_ops.get(key[0])
                if resource is not None:
                    choice = arbitrate(resource, candidates)
                    if choice is None or key != choice[0]:
                        record_block(key, f"arbitration:{resource}")
                        continue
                    arbiters[resource].grant(choice)
                    grants[resource].append(dict(op=key[0], iteration=key[1], kind=op.kind, start=current))
                placed[key] = current
                # A failed attempt followed by a grant in the same tick did
                # not spend a clock blocked (e.g. the second pooled port).
                for spans in blocked.get(key, {}).values():
                    if spans and spans[-1][1] > current:
                        spans[-1][1] = current
                        if spans[-1][0] == current:
                            spans.pop()
                for t in issue_ticks(op, current):
                    issued[op.domain][t].append((key, op.kind))
                del ready[key]
                progress = True
                for use in op.uses:
                    for tick in range(current + use.offset, current + use.offset + use.duration):
                        calendar[use.resource][tick] += use.rate
                if op.kind in windows and op.window_domain is not None:
                    active[op.window_domain].append((current, windows[op.kind]))
                for dst, delay in successors[key]:
                    indegree[dst] -= 1
                    earliest[dst] = max(earliest[dst], current + delay)
                    if indegree[dst] == 0:
                        ready[dst] = earliest[dst]
            if not progress:
                current += 1
                for domain in active:
                    active[domain] = [(t, pat) for t, pat in active[domain] if t + len(pat) > current]
        if len(placed) != len(keys):
            raise ValueError("expanded graph contains a cycle")
        # Independently check the generated witness before reporting it.
        for src, dst, delay, _ in expanded_edges:
            if placed[dst] < placed[src] + delay:
                raise AssertionError("schedule violates a dependency")
        for resource, values in calendar.items():
            if max(values.values(), default=0) > capacities[resource] + 1e-8:
                raise AssertionError("schedule exceeds a resource capacity")
        for key, start in placed.items():
            op = ops[key[0]]
            if op.kind in windows and op.window_domain is not None:
                pattern = windows[op.kind]
                for offset, allowed in enumerate(pattern):
                    for other, kind in issued[op.window_domain][start + offset]:
                        if other != key and kind not in allowed and "*" not in allowed:
                            raise AssertionError("schedule violates a coexecution window")
        events = [
            dict(
                op=name, iteration=i, phase=ops[name].phase, kind=ops[name].kind, start=start,
                end=start + ops[name].latency, reservations=[
                    dict(resource=u.resource, start=start + u.offset, end=start + u.offset + u.duration, rate=u.rate,
                         work=u.work) for u in ops[name].uses
                ]) for (name, i), start in placed.items() if name in original_ops
        ]
        events.sort(key=lambda e: (e["start"], e["iteration"], e["op"]))
        compute = set(self.data.get("compute_resources", []))
        anchors = []
        for i in range(iterations):
            starts = [
                u["start"] for e in events if e["iteration"] == i for u in e["reservations"] if u["resource"] in compute
            ]
            if starts:
                anchors.append(min(starts))
        result = dict(iterations=iterations, makespan=max(e["end"] for e in events), events=events,
                      policy=f"{policy} list scheduling", priority_target_period=target_period, storage=storage,
                      coexecution="explicit issue windows" if windows else "unconstrained", verified=True,
                      finite_dependency_lower_bound=finite_dependency_bound,
                      finite_resource_lower_bound=finite_resource_bound,
                      finite_lower_bound=max(finite_dependency_bound, finite_resource_bound))
        result["phase_windows"] = {
            phase: [
                min(e["start"]
                    for e in events
                    if e["phase"] == phase),
                max(e["end"]
                    for e in events
                    if e["phase"] == phase)
            ]
            for phase, members in phases.items()
            if members
        }
        result["compute_service_ticks"] = sum(use["work"]
                                              for e in events
                                              for use in e["reservations"]
                                              if use["resource"] in compute)
        if compute:
            result["dispatch_compute_utilization"] = result["compute_service_ticks"] / (result["makespan"] *
                                                                                        sum(capacities[r]
                                                                                            for r in compute))
        result["queue_occupancy"] = self.queue_occupancy(placed, original_ops, iterations)
        result["admissions"] = self.admission_diagnostics(placed, original_ops, expanded_edges, iterations)
        result["scheduling_blocks"] = [
            dict(op=name, iteration=i, reasons={reason: spans
                                                for reason, spans in reasons.items()
                                                if spans})
            for (name, i), reasons in blocked.items()
            if any(reasons.values())
        ]
        result["arbitration"] = {
            resource: dict(spec, grants=grants[resource], quantum="one admitted operation, independent of its bytes")
            for resource, spec in self.arbitration.items()
        }
        result["waits"], coverage = self.wait_diagnostics(placed, original_ops, iterations)
        result["requests"] = self.request_diagnostics(placed, original_ops, iterations, coverage, calendar)
        result["buffer_lifetimes"] = self.buffer_lifetimes(placed, original_ops, iterations)
        result["issue_windows"] = self.issue_windows(events, capacities)
        result["periodic_witness"] = self.periodic_witness(placed, capacities, original_ops, edges, iterations, warmup)
        if self.arbitration:
            result["periodic_witness_limit"] = "stateful arbitration is checked only in the finite schedule"
        if len(anchors) == iterations and iterations - warmup > warmup:
            lo, hi = anchors[warmup], anchors[iterations - warmup - 1]
            if hi > lo:
                utilization = {
                    r: sum(v
                           for t, v in cal.items()
                           if lo <= t < hi) / (capacities[r] * (hi - lo))
                    for r, cal in calendar.items()
                }
                result.update(
                    interior_window=[lo, hi], interior_periods=iterations - 2 * warmup - 1,
                    interior_cycles_per_period=(hi - lo) / (iterations - 2 * warmup - 1),
                    interior_resource_utilization=utilization,
                    interior_compute_utilization=sum(utilization[r] * capacities[r]
                                                     for r in compute) / sum(capacities[r] for r in compute))
        selected_windows = dict(dispatch=[0, result["makespan"]], **result["phase_windows"])
        if "interior_window" in result:
            selected_windows["interior"] = result["interior_window"]
        for name, span in (analysis_windows or {}).items():
            if name in selected_windows:
                raise ValueError(f"analysis window name is reserved: {name}")
            selected_windows[name] = span
        result["windows"] = self.schedule_windows(result, capacities, placed, original_ops, iterations,
                                                  selected_windows, calendar)
        return result

    @staticmethod
    def instance_pairs(source, target, distance, ops, iterations):
        src, dst = ops[source].phase, ops[target].phase
        if src == dst == "loop":
            return [(i, i + distance) for i in range(iterations - distance)]
        if src == "loop":
            return [(iterations - 1, -1)]
        if dst == "loop":
            return [(-1, 0)]
        return [(-1, -1)]

    @staticmethod
    def distribution(values):
        return dict(count=len(values), minimum=min(values) if values else None,
                    mean=sum(values) / len(values) if values else None, maximum=max(values) if values else None)

    def wait_diagnostics(self, placed, ops, iterations):
        """Report completion joins, without charging a slow peer twice.

        The interval is measured from the declared arrival. It is a property
        of this schedule, not a sum of independently attributable stall causes.
        """
        rows, coverage = [], defaultdict(list)
        for wait in self.waits:
            points = []
            for raw in [wait["arrival"], *wait["completions"]]:
                point = self.endpoint(raw)
                delay = ticks(self.value(point["delay"])) if "delay" in point else ops[point["op"]].latency
                times = {
                    dst: (src, placed[point["op"], src] + delay)
                    for src, dst in self.instance_pairs(point["op"], wait["target"], point.get("distance", 0), ops,
                                                        iterations)
                }
                points.append((point, times))
            for i in (range(iterations) if ops[wait["target"]].phase == "loop" else [-1]):
                arrival = points[0][1].get(i)
                if arrival is None:
                    continue
                complete = [(point["op"], times[i][0], times[i][1]) for point, times in points[1:] if i in times]
                ready = max([arrival[1]] + [t for _, _, t in complete])
                start = placed[wait["target"], i]
                assert start >= ready
                row = dict(wait=wait["id"], iteration=i, role=wait.get("role", "completion join"), arrival=arrival[1],
                           compute_resources=wait.get("compute_resources", []), ready=ready, start=start,
                           end=start + ops[wait["target"]].latency, completion_wait_ticks=ready - arrival[1],
                           scheduling_delay_ticks=start - ready, covered_requests=len(complete),
                           last_completions=[dict(op=name, iteration=j) for name, j, t in complete if t == ready])
                rows.append(row)
                for name, j, _ in complete:
                    coverage[name, j].append(row)
        return rows, coverage

    def request_diagnostics(self, placed, ops, iterations, coverage, calendar):
        rows = []
        direct_delays = {}
        for edge in self.raw_edges:
            if not edge.get("distance", 0):
                pair = (edge["source"], edge["target"])
                delay = ticks(self.value(edge["delay"])) if "delay" in edge else ops[edge["source"]].latency
                direct_delays[pair] = max(direct_delays.get(pair, 0), delay)
        for request in self.requests:
            issue, complete = request["issue"], request["complete"]
            route = list(dict.fromkeys([issue, *request.get("stages", []), complete]))
            pairs = list(zip(route, route[1:]))
            route_complete = not request.get("aggregate", False) and all(pair in direct_delays for pair in pairs)
            for i in (range(iterations) if ops[issue].phase == "loop" else [-1]):
                begin, end = placed[issue, i], placed[complete, i] + ops[complete].latency
                route_queue = (sum(max(0, placed[b, i] - placed[a, i] - direct_delays[a, b])
                                   for a, b in pairs) if route_complete else None)
                consumer_times = []
                for raw in request.get("consumers", []):
                    point = self.endpoint(raw)
                    consumer_times.extend(placed[point["op"], dst] + ticks(self.value(point.get("delay", 0)))
                                          for src, dst in self.instance_pairs(issue, point["op"],
                                                                              point.get("distance", 0), ops, iterations)
                                          if src == i)
                waits = coverage.get((complete, i), [])
                first_wait = min(waits, key=lambda w: w["arrival"]) if waits else None
                row = dict(
                    request=request["id"], iteration=i, phase=ops[issue].phase, kind=request.get("kind", "memory"),
                    scope=request.get("scope", "unspecified"), issue=begin, complete=end,
                    bytes=self.value(request["bytes"]) if "bytes" in request else None, latency=end - begin,
                    route_queue_ticks=route_queue, route_complete=route_complete,
                    aggregate=request.get("aggregate", False), stages=[
                        dict(op=name, start=placed[name, i], end=placed[name, i] + ops[name].latency) for name in route
                    ], first_consumer_start=min(consumer_times) if consumer_times else None,
                    first_covering_wait=first_wait["wait"] if first_wait else None,
                    first_wait_arrival=first_wait["arrival"] if first_wait else None,
                    lead_to_first_wait=first_wait["arrival"] - begin if first_wait else None,
                    completion_after_first_wait=max(0, end - first_wait["arrival"]) if first_wait else None)
                if "source_release" in request:
                    point = self.endpoint(request["source_release"])
                    delay = ticks(self.value(point["delay"])) if "delay" in point else ops[point["op"]].latency
                    row["source_release"] = placed[point["op"], i] + delay
                    if row["source_release"] > end:
                        raise ValueError(f"request source remains live after result completion: {request['id']}")
                if consumer_times and min(consumer_times) < end:
                    raise ValueError(f"request consumed before completion: {request['id']}")
                resource = request.get("compute_resource")
                if resource and first_wait:
                    row["compute_resource"] = resource
                    row["matrix_service_before_first_wait"] = sum(
                        calendar[resource].get(t, 0) for t in range(begin, max(begin, first_wait["arrival"])))
                rows.append(row)
        return dict(
            events=rows, by_kind={
                kind:
                dict(
                    latency=self.distribution([r["latency"]
                                               for r in rows
                                               if r["kind"] == kind]), route_queue_ticks=self.distribution([
                                                   r["route_queue_ticks"]
                                                   for r in rows
                                                   if r["kind"] == kind and r["route_queue_ticks"] is not None
                                               ]))
                for kind in sorted({r["kind"]
                                    for r in rows})
            })

    def buffer_lifetimes(self, placed, ops, iterations):
        rows = []
        for buf in self.buffers:
            producer, slots = buf["producer"], int(self.value(buf["slots"]))
            lifetimes, slacks = [], []
            for i in range(iterations):
                release = max(placed[c, i] +
                              (ticks(self.value(buf["release_delay"])) if "release_delay" in buf else ops[c].latency)
                              for c in buf["consumers"])
                lifetimes.append(release - placed[producer, i])
                if i + slots < iterations:
                    slack = placed[producer, i + slots] - release
                    assert slack >= 0, "buffer overwritten before its declared retirement"
                    slacks.append(slack)
            rows.append(
                dict(buffer=buf["id"], slots=slots, lifetime=self.distribution(lifetimes),
                     reuse_slack=self.distribution(slacks)))
        return rows

    def issue_windows(self, events, capacities):
        """Count permitted empty issue slots; readiness is not implied."""
        windows = self.data.get("coexecution", {})
        allowed = defaultdict(dict)
        used = defaultdict(lambda: defaultdict(float))
        classes = {e["kind"] for e in events}
        for event in events:
            op = self.raw_ops[event["op"]]
            domain = op.get("domain")
            for use in event["reservations"]:
                if use["resource"] == domain:
                    for tick in range(use["start"], use["end"]):
                        used[domain][tick] += use["rate"]
            window_domain = op.get("window_domain", domain)
            if window_domain is not None:
                for offset, kinds in enumerate(windows.get(event["kind"], [])):
                    tick = event["start"] + offset
                    permitted = classes if "*" in kinds else set(kinds)
                    allowed[window_domain][tick] = allowed[window_domain].get(tick, classes) & permitted
        return [
            dict(
                domain=domain, active_ticks=len(timeline), kinds={
                    kind:
                    dict(
                        permitted_ticks=sum(kind in kinds
                                            for kinds in timeline.values()),
                        empty_permitted_ticks=sum(kind in kinds and used[domain][tick] == 0
                                                  for tick, kinds in timeline.items()))
                    for kind in sorted(classes)
                },
                interpretation="empty permitted slots; dependencies, credits and register readiness may prevent issue")
            for domain, timeline in allowed.items()
        ]

    def admission_diagnostics(self, placed, ops, edges, iterations):
        """Separate circular-credit readiness from subsequent issue delay.

        The frontier includes other dependencies and admission ordering. These
        are local delays in the supplied schedule, not a counterfactual run
        with a queue removed. Multiple pools can block the same acquisition.
        """
        frontiers = defaultdict(int)
        credit_ready = defaultdict(lambda: defaultdict(int))
        for source, target, delay, reason in edges:
            time = placed[source] + delay
            if reason.startswith("queue-reuse:"):
                name = reason.removeprefix("queue-reuse:")
                credit_ready[target][name] = max(credit_ready[target][name], time)
            else:
                frontiers[target] = max(frontiers[target], time)
        rows = []
        for queue in self.queues:
            for index, entry in enumerate(queue["entries"]):
                name = entry["acquire"]
                for iteration in (range(iterations) if ops[name].phase == "loop" else [-1]):
                    key = name, iteration
                    arrival = frontiers[key]
                    ready = max(arrival, credit_ready[key].get(queue["id"], 0))
                    all_ready = max([arrival, *credit_ready[key].values()])
                    admitted = placed[key]
                    assert admitted >= all_ready
                    rows.append(
                        dict(queue=queue["id"], entry=index, op=name, iteration=iteration, arrival=arrival,
                             credit_ready=ready, admitted=admitted, credit_wait_ticks=ready - arrival,
                             scheduling_delay_ticks=admitted - all_ready))
        return rows

    def schedule_windows(self, schedule, capacities, placed, ops, iterations, windows, calendar):
        """Compare pressure, loaded latency and idle overlap over one clock window."""
        compute = self.data.get("compute_resources", [])
        op_scope = defaultdict(set)
        for request in self.requests:
            if "compute_resource" in request:
                names = [request["issue"], request["complete"], *request.get("stages", [])]
                names += [self.endpoint(p)["op"] for p in request.get("consumers", [])]
                for name in names:
                    op_scope[name].add(request["compute_resource"])
        for wait in self.waits:
            op_scope[wait["target"]].update(wait.get("compute_resources", []))
        intervals = defaultdict(lambda: defaultdict(list))
        for wait in schedule["waits"]:
            for resource in wait["compute_resources"]:
                intervals[resource]["completion-wait:" + wait["role"]].append((wait["arrival"], wait["ready"]))
        for admission in schedule["admissions"]:
            for resource in op_scope[admission["op"]]:
                intervals[resource]["credit:" + admission["queue"]].append(
                    (admission["arrival"], admission["credit_ready"]))
        for row in schedule["scheduling_blocks"]:
            for resource in op_scope[row["op"]]:
                for reason, spans in row["reasons"].items():
                    intervals[resource][reason].extend(spans)

        result = {}
        for name, span in windows.items():
            if (not isinstance(span, (list, tuple)) or len(span) != 2 or any(type(t) is not int for t in span)
                    or not 0 <= span[0] <= span[1] <= schedule["makespan"]):
                raise ValueError(f"invalid analysis window: {name}={span}")
            lo, hi = span
            if lo == hi:
                continue

            def union(spans):
                merged = []
                for start, end in sorted((max(lo, a), min(hi, b)) for a, b in spans if a < hi and b > lo):
                    if start >= end:
                        continue
                    if merged and start <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], end)
                    else:
                        merged.append([start, end])
                return merged

            def length(spans):
                return sum(b - a for a, b in union(spans))

            matrix = {}
            for resource in compute:

                def idle_overlap(spans):
                    return sum(
                        max(0, capacities[resource] - calendar[resource].get(t, 0))
                        for a, b in union(spans)
                        for t in range(a, b))

                service = sum(calendar[resource].get(t, 0) for t in range(lo, hi))
                idle = capacities[resource] * (hi - lo) - service
                overlap = {reason: idle_overlap(spans) for reason, spans in intervals[resource].items()}
                total = idle_overlap([span for spans in intervals[resource].values() for span in spans])
                matrix[resource] = dict(service_ticks=service, idle_ticks=idle,
                                        idle_overlap_by_reason={k: v
                                                                for k, v in overlap.items()
                                                                if v}, idle_overlap_union_ticks=total,
                                        unclassified_idle_ticks=max(0, idle - total))
            requests = [r for r in schedule["requests"]["events"] if lo < r["complete"] <= hi]
            queues = self.queue_occupancy(placed, ops, iterations, (lo, hi))
            by_queue = defaultdict(list)
            for row in schedule["admissions"]:
                by_queue[row["queue"]].append(row)
            for queue in queues:
                admissions = by_queue[queue["queue"]]
                queue["credit_wait_ticks"] = length([(row["arrival"], row["credit_ready"]) for row in admissions])
                queue["credit_wait_event_ticks"] = sum(
                    max(0,
                        min(hi, row["credit_ready"]) - max(lo, row["arrival"])) for row in admissions)
            numerator = sum(row["service_ticks"] for row in matrix.values())
            denominator = (hi - lo) * sum(capacities[r] for r in compute)
            result[name] = dict(
                window=[lo, hi], matrix_resources=matrix, service_ticks=numerator, available_resource_ticks=denominator,
                service_fraction=numerator / denominator if denominator else None, queues=queues, requests={
                    kind:
                    dict(
                        latency=self.distribution([r["latency"]
                                                   for r in requests
                                                   if r["kind"] == kind]), route_queue_ticks=self.distribution([
                                                       r["route_queue_ticks"]
                                                       for r in requests
                                                       if r["kind"] == kind and r["route_queue_ticks"] is not None
                                                   ]))
                    for kind in schedule["requests"]["by_kind"]
                },
                interpretation="service/occupancy use [start,end); latency samples complete in (start,end] and retain "
                "their full lifetime. Idle overlaps are not causal attribution; categories may overlap.")
        return result

    def queue_occupancy(self, placed, ops, iterations, window=None):
        """Independently validate finite credit occupancy from the witness."""
        rows = []
        for queue in self.queues:
            changes, lifetimes, retirements = defaultdict(float), [], []
            acquired = released = 0
            lo, hi = window if window is not None else (0, math.inf)
            for index, entry in enumerate(queue["entries"]):
                phase = ops[entry["acquire"]].phase
                delay = ticks(self.value(
                    entry["release_delay"])) if "release_delay" in entry else ops[entry["release"]].latency
                for iteration in (range(iterations) if phase == "loop" else [-1]):
                    begin = placed[entry["acquire"], iteration]
                    end = placed[entry["release"], iteration] + delay
                    if end < begin:
                        raise AssertionError("queue releases credits before their acquisition")
                    units = self.value(entry.get("units", 1))
                    acquired += lo <= begin < hi
                    released += lo < end <= hi
                    if begin < hi and end > lo:
                        changes[max(lo, begin)] += units
                        changes[min(hi, end)] -= units
                        lifetimes.append(end - begin)
                    elif window is None:
                        lifetimes.append(end - begin)
                    retirements.append((iteration, index, end))
            if queue.get("ordered_retirement", False):
                ends = [end for _, _, end in sorted(retirements)]
                if any(a > b for a, b in zip(ends, ends[1:])):
                    raise AssertionError(f"queue {queue['id']} retires entries out of order")
            occupied = peak = area = saturated = previous = 0
            capacity = self.value(queue["capacity"])
            for time, delta in sorted(changes.items()):
                area += occupied * (time - previous)
                if occupied == capacity:
                    saturated += time - previous
                previous = time
                occupied += delta
                peak = max(peak, occupied)
                if occupied < -1e-8 or occupied > capacity + 1e-8:
                    raise AssertionError(f"queue {queue['id']} exceeds its credit capacity")
            rows.append(
                dict(queue=queue["id"], capacity=capacity, peak_credits=peak, scope=queue.get("scope", "unspecified"),
                     unit=queue.get("unit",
                                    "credits"), credit_ticks=area, saturated_ticks=saturated, acquisitions=acquired,
                     releases=released, ordered_retirement=queue.get("ordered_retirement", False),
                     minimum_lifetime=min(lifetimes) if lifetimes else None,
                     maximum_lifetime=max(lifetimes) if lifetimes else None,
                     mean_lifetime=sum(lifetimes) / len(lifetimes) if lifetimes else None))
        return rows


def plot_schedule(schedule, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resources = sorted({u["resource"] for e in schedule["events"] for u in e["reservations"]})
    lo, hi = schedule.get("interior_window", [0, schedule["makespan"]])
    hi = min(hi, lo + 3 * schedule.get("interior_cycles_per_period", schedule["makespan"]))
    colors = {
        kind: plt.get_cmap("tab10")(i % 10)
        for i, kind in enumerate(sorted({e["kind"]
                                         for e in schedule["events"]}))
    }
    fig, ax = plt.subplots(figsize=(12, max(3, len(resources) * 0.38)))
    for e in schedule["events"]:
        for use in e["reservations"]:
            start, end = max(lo, use["start"]), min(hi, use["end"])
            if end > start:
                ax.broken_barh([(start, end - start)], (resources.index(use["resource"]) - 0.35, 0.7),
                               facecolors=colors[e["kind"]], alpha=0.8)
    ax.set_yticks(range(len(resources)), resources)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Model clock ticks (see report for calibration and assumptions)")
    ax.set_title("Constructive overlap schedule: resource reservations")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def print_summary(report):
    print(f"{report['name']} ({report['clock']})")
    print(f"{'resource':<20} {'work/period':>24} {'capacity':>14} {'min ticks':>12}")
    for row in report["resources"]:
        capacity = f"{row['capacity']:g}" if "capacity" in row else str(row['capacity_expression'])
        cycles = f"{row['cycles']:.3f}" if "cycles" in row else "unknown"
        print(f"{row['resource']:<20} {row['work_expression']:>24} {capacity:>14} {cycles:>12}")
    qualifier = "partial" if report["bound_is_partial"] else "resource and recurrence constraints"
    print(f"Period lower bound: {report['period_lower_bound']:.3f} ticks ({qualifier})")
    if "compute_utilization_upper_bound" in report:
        print(f"Optimistic compute utilization cap: {100 * report['compute_utilization_upper_bound']:.3f}%")
    if report["unresolved"]:
        print("Unfilled parameters: " + ", ".join(report["unresolved"]))
    for space, allocation in report.get("storage", {}).items():
        if allocation["fits"] is False:
            print(f"Infeasible storage allocation: {space} needs {allocation['required']:g}, "
                  f"reserves {allocation['allocated']:g} {allocation['unit']}; "
                  f"capacity is {allocation['capacity']}")
    residency = report.get("residency", {})
    if residency.get("constraints"):
        print(
            f"Residency upper bound from known capacities: {residency['upper_bound_from_known_constraints']} CTA(s); "
            f"capacity inputs {'resolved' if residency['all_constraints_resolved'] else 'partial'}, admission not inferred"
        )
    if "schedule" in report:
        schedule = report["schedule"]
        print(f"Constructive schedule: {schedule['makespan']} ticks for {schedule['iterations']} iterations and drain")
        print(f"Finite-dispatch lower bound: {schedule['finite_lower_bound']:.3f} ticks")
        for kind, measurements in schedule["requests"]["by_kind"].items():
            latency, queued = measurements["latency"], measurements["route_queue_ticks"]
            queued_text = (f"{queued['mean']:.2f} ticks"
                           if queued["count"] else "unavailable (aggregate or incomplete route)")
            print(f"{kind} requests: {latency['count']}, scheduled latency min/mean/max "
                  f"{latency['minimum']}/{latency['mean']:.2f}/{latency['maximum']} ticks; "
                  f"mean queueing between stages {queued_text}")
        if schedule["waits"]:
            print(
                f"Longest declared completion wait: {max(w['completion_wait_ticks'] for w in schedule['waits'])} ticks "
                "(wait intervals can overlap)")
        interior = schedule["windows"].get("interior")
        if interior is not None and interior["service_fraction"] is not None:
            print(f"Interior window {interior['window']}: {100 * interior['service_fraction']:.3f}% compute service")
            for kind, measurements in interior["requests"].items():
                latency = measurements["latency"]
                if latency["count"]:
                    print(f"  {kind}: {latency['count']} completions, mean loaded latency {latency['mean']:.2f} ticks")
            for queue in sorted(interior["queues"], key=lambda q: -q["credit_wait_ticks"])[:3]:
                if queue["credit_wait_ticks"]:
                    print(f"  {queue['queue']}: {queue['credit_wait_ticks']} credit-wait ticks, "
                          f"{queue['saturated_ticks']} full ticks (overlapping measurements)")
        witness = schedule["periodic_witness"]
        if witness:
            print(f"Verified repeating schedule: {witness['cycles']} ticks / {witness['iterations']} iterations "
                  f"= {witness['cycles_per_iteration']:.3f} ticks/iteration")
            if witness["compute_utilization"] is not None:
                print(f"Compute utilization in that schedule: {100 * witness['compute_utilization']:.3f}%")
        else:
            print("No repeating schedule certified; finite-window measurements are in the JSON report.")
            if "periodic_witness_limit" in schedule:
                print("  " + schedule["periodic_witness_limit"])


def print_sweep(reports, dimensions):
    names = [dim[0][0] for dim in dimensions]
    print(",".join(names + ["period_lower_bound", "utilization_cap", "verified_period", "verified_utilization"]))
    for report in reports:
        witness = report.get("schedule", {}).get("periodic_witness")
        values = [report["parameter_values"][name] for name in names]
        values += [
            report["period_lower_bound"],
            report.get("compute_utilization_upper_bound", ""), witness["cycles_per_iteration"] if witness else "",
            witness["compute_utilization"] if witness else ""
        ]
        print(",".join(str(value) for value in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--bindings", type=Path, help="JSON object of measured or hypothetical parameter values")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--sweep", action="append", default=[], metavar="NAME=VALUE,VALUE",
                        help="repeat for a Cartesian sweep; all values remain scenario assumptions")
    parser.add_argument("--target-period", type=float,
                        help="evaluate bandwidth load and required ring depth at this period")
    parser.add_argument("--schedule", action="store_true",
                        help="requires all used capacities and latencies to be supplied")
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--max-cycles", type=int, default=1000000)
    parser.add_argument("--policy", choices=("critical-path", "deadline"), default="critical-path",
                        help="list-scheduling priority; deadline uses ideal compute dates, not hard timing constraints")
    parser.add_argument("--window", action="append", default=[], metavar="NAME=START:END",
                        help="additional clock window for pressure and latency diagnostics; requires --schedule")
    parser.add_argument("--json", type=Path, help="write the report, including every scheduled operation")
    parser.add_argument("--print-json", action="store_true", help="print report metadata instead of the compact table")
    parser.add_argument("--plot", type=Path, help="write a resource timeline (requires matplotlib and --schedule)")
    args = parser.parse_args()
    try:
        bindings = json.loads(args.bindings.read_text()) if args.bindings else {}
        overrides = {}
        for item in args.set:
            key, value = item.split("=", 1)
            overrides[key] = float(value)
        analysis_windows = {}
        for item in args.window:
            key, value = item.split("=", 1)
            if not key or key in analysis_windows:
                raise ValueError("analysis window names must be nonempty and unique")
            analysis_windows[key] = [int(v) for v in value.split(":")]
        if analysis_windows and not args.schedule:
            raise ValueError("--window requires --schedule")
        if args.target_period is not None and args.target_period <= 0:
            raise ValueError("target-period must be positive")
        data = json.loads(args.model.read_text())
        model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
        tool_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        dimensions = []
        for item in args.sweep:
            key, values = item.split("=", 1)
            dimensions.append([(key, float(value)) for value in values.split(",")])
        reports = []
        for values in product(*dimensions):
            model = Model(data, bindings, {**overrides, **dict(values)})
            report = model.bounds(args.target_period)
            report["model_file"] = str(args.model)
            report["model_sha256"] = model_hash
            report["tool_sha256"] = tool_hash
            report["bindings_file"] = str(args.bindings) if args.bindings else None
            report["bindings_sha256"] = hashlib.sha256(
                args.bindings.read_bytes()).hexdigest() if args.bindings else None
            report["overrides"] = {**overrides, **dict(values)}
            if args.schedule:
                report["schedule"] = model.schedule(args.iterations, args.warmup, args.max_cycles, args.policy,
                                                    args.target_period or report["period_lower_bound"],
                                                    analysis_windows)
            reports.append(report)
        report = reports[0] if not dimensions else {"scenarios": reports}
        if args.plot:
            if not args.schedule or dimensions:
                raise ValueError("--plot requires --schedule and a single scenario")
            plot_schedule(report["schedule"], args.plot)
        if args.json:
            args.json.write_text(json.dumps(report, indent=2) + "\n")
        summaries = []
        for item in reports:
            summary = dict(item)
            if "schedule" in item:
                summary["schedule"] = {k: v for k, v in item["schedule"].items() if k != "events"}
            summaries.append(summary)
        summary = summaries[0] if not dimensions else {"scenarios": summaries}
        if args.print_json:
            print(json.dumps(summary, indent=2))
        elif dimensions:
            print_sweep(reports, dimensions)
        else:
            print_summary(report)
    except (ValueError, KeyError, ZeroDivisionError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
