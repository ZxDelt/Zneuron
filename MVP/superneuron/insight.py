"""Reading the network: why a neuron fired, what it learned, what it is "thinking", and alarms."""

from collections import defaultdict


def last_fire(net, nid):
    for t, k in reversed(net.fire_log):
        if k == nid:
            return t
    return None


def why_fired(net, nid, t=None, depth=4, breadth=4):
    """Tree of what built up the charge for a fire, traced back through earlier fires."""
    t = last_fire(net, nid) if t is None else t
    if t is None:
        return None
    causes = net.fire_causes.get((t, nid))
    node = {"id": nid, "t": t, "sensor": net.neurons[nid].sensor, "causes": []}
    if causes is None:
        node["forgotten"] = True            # older than what the network keeps
        return node
    strongest = sorted(causes, key=lambda c: -abs(c[2]))[:breadth]
    for src, arrive_t, charge in strongest:
        if src == -1:
            node["causes"].append({"outside": True, "t": arrive_t, "charge": round(charge, 3)})
            continue
        line = net.line_between(src, nid)
        send_t = arrive_t - (line.delay if line else net.p.delay_ms)
        child = {"id": src, "t": send_t, "charge": round(charge, 3), "causes": []}
        if depth > 1:
            deeper = why_fired(net, src, send_t, depth - 1, breadth)
            if deeper:
                child.update({k: v for k, v in deeper.items() if k not in ("charge",)})
        node["causes"].append(child)
    return node


def chains(net, window_ms=60, limit=6):
    """Spike paths that happened recently, e.g. [1, 7, 22]: 1 caused 7, 7 caused 22."""
    start = net.t - window_ms
    recent = [(t, k) for t, k in net.fire_log if t >= start]
    found = []
    for t, k in reversed(recent):
        path = [k]
        cur_t, cur = t, k
        while len(path) < 8:
            causes = [c for c in net.fire_causes.get((cur_t, cur), []) if c[0] != -1 and c[2] > 0]
            if not causes:
                break
            src, arrive_t, _ = max(causes, key=lambda c: c[2])
            line = net.line_between(src, cur)
            cur_t, cur = arrive_t - (line.delay if line else net.p.delay_ms), src
            if cur_t < start or cur in path:
                break
            path.append(cur)
        if len(path) >= 2:
            path.reverse()
            # keep the longest version of each route: drop routes this one extends
            if not any(f[: len(path)] == path for f in found):
                found = [f for f in found if path[: len(f)] != f]
                found.append(path)
        if len(found) >= limit:
            break
    return sorted(found, key=len, reverse=True)


class GroupTracker:
    """Finds groups of neurons that keep firing together and keeps their names stable over time."""

    def __init__(self):
        self.groups = {}       # name -> set of neuron ids
        self.next_id = 1

    def update(self, net, window_ms=500, gap_ms=5, min_together=3, min_size=3):
        start = net.t - window_ms
        fires = sorted((t, k) for t, k in net.fire_log if t >= start)
        together = defaultdict(int)
        for i, (t, a) in enumerate(fires):
            k = i + 1
            while k < len(fires) and fires[k][0] - t <= gap_ms:
                b = fires[k][1]
                if b != a:
                    together[(min(a, b), max(a, b))] += 1
                k += 1
        links = defaultdict(set)
        for (a, b), n in together.items():
            if n >= min_together:
                links[a].add(b)
                links[b].add(a)
        seen, found = set(), []
        for start_id in links:
            if start_id in seen:
                continue
            stack, comp = [start_id], set()
            while stack:
                x = stack.pop()
                if x in comp:
                    continue
                comp.add(x)
                stack.extend(links[x] - comp)
            seen |= comp
            if len(comp) >= min_size:
                found.append(comp)

        named = {}
        free = dict(self.groups)
        for comp in sorted(found, key=len, reverse=True):
            best, score = None, 0.0
            for name, old in free.items():
                s = len(comp & old) / len(comp | old)
                if s > score:
                    best, score = name, s
            if best and score >= 0.4:
                free.pop(best)
                named[best] = comp
            else:
                named[f"G{self.next_id}"] = comp
                self.next_id += 1
        self.groups = named
        return [{"name": k, "members": sorted(v)} for k, v in sorted(named.items(), key=lambda kv: -len(kv[1]))]


def capacity(net, groups):
    n = net.alive_count()
    slots = n * net.p.max_lines / 2          # each line uses a slot on two neurons
    return {
        "awake": len(net.awake), "neurons": n,
        "lines": len(net.lines), "line_slots": int(slots),
        "groups": len(groups), "max_groups": max(1, n // 3),
    }


def learned(net, window_ms=2000, top=6):
    start = net.t - window_ms
    total = defaultdict(float)
    for t, s, d, change in net.learn_log:
        if t >= start:
            total[(s, d)] += change
    ranked = sorted(total.items(), key=lambda kv: kv[1])
    down = [[s, d, round(v, 4)] for (s, d), v in ranked[:top] if v < 0]
    up = [[s, d, round(v, 4)] for (s, d), v in reversed(ranked[-top:]) if v > 0]
    events = [list(e) for e in net.events if e[0] >= net.t - 10_000][-12:]
    return {"up": up, "down": down, "events": events}


def tell(net, groups, alarm_list):
    """Brain-state teller: the current state in plain sentences (a readout, not the brain talking)."""
    t = net.t
    recent = {k for tt, k in net.fire_log if tt > t - 1000}
    sensors = [n for n in net.neurons if n.sensor]
    motors = [n for n in net.neurons if n.motor]
    inner = [n for n in net.neurons if not n.sensor and n.alive]
    lines = [f"Time {t / 1000:.1f} s. {len(net.awake)} of {net.alive_count()} neurons awake; "
             f"{len(recent)} fired in the last second."]
    on = [n.id for n in sensors if n.id in recent]
    lines.append(f"Senses: {len(on)} of {len(sensors)} sensor neurons active." if sensors else "Senses: none connected.")
    if motors:
        moving = [n.id for n in motors if n.id in recent]
        lines.append(f"Motors: {len(moving)} of {len(motors)} firing — the body is "
                     f"{'moving' if moving else 'still'}." + (f" Active: {moving[:10]}" if moving else ""))
    if groups:
        lines.append("Thinking: " + "; ".join(f"{g['name']} ({len(g['members'])} neurons)" for g in groups[:4]) + ".")
    else:
        lines.append("Thinking: no group is firing together right now.")
    start = t - 2000
    ups = sum(1 for tt, *_ , c in net.learn_log if tt >= start and c > 0)
    downs = sum(1 for tt, *_ , c in net.learn_log if tt >= start and c < 0)
    lines.append(f"Learning: {ups} strengthen and {downs} weaken steps in the last 2 s; "
                 f"{net.counts['grow']} lines grown, {net.counts['prune']} pruned, {net.counts['lock']} locked, "
                 f"{net.counts['birth']} neurons born, {net.counts['death']} died so far.")
    cf = sum(n.cfuel for n in inner) / max(1, len(inner))
    lines.append(f"Drives: Lfuel {net.lfuel:+.1f} (reward), Rfuel {net.rfuel:.1f} (calm), curiosity {cf:.2f}.")
    for level, msg in alarm_list:
        if level != "ok":
            lines.append(f"⚠ {msg}")
    return lines


def alarms(net):
    out = []
    inner = [n for n in net.neurons if not n.sensor]
    now = sum(1 for t, k in net.fire_log if t >= net.t - 2 and not net.neurons[k].sensor)
    if inner and now / len(inner) > 0.2:
        out.append(["danger", f"Runaway: {now} of {len(inner)} neurons fired in the last 2 ms (F2 limit 20%)."])
    last_inner = max((t for t, k in net.fire_log if not net.neurons[k].sensor), default=None)
    if net.t > 2000 and (last_inner is None or net.t - last_inner > 2000):
        out.append(["warn", "Silent: no non-sensor neuron has fired for 2 s."])
    minis = [m for line in net.lines.values() for m in (line.ab, line.ba) if m.active]
    locked = sum(m.locked for m in minis)
    if minis and locked / len(minis) > 0.4:
        out.append(["warn", f"Memory freezing: {locked} of {len(minis)} lines are locked."])
    if inner and sum(n.rfuel_local for n in inner) / len(inner) >= 1.9:
        out.append(["warn", "Glia are calming everything (local Rfuel near max 2)."])
    if not out:
        out.append(["ok", "No problems detected."])
    return out
