"""All numbers from neuron_spec.json, in one place. Spec item IDs in comments."""

from dataclasses import dataclass


@dataclass
class Params:
    # Clock (F6): one 1 ms tick for awake neurons, sleeping neurons skip work
    tick_ms: int = 1

    # Charge and firing
    hub_cap: float = 1.0               # D4
    thr_start: float = 0.5             # homeostasis threshold starts at the old 0.5
    thr_min: float = 0.3               # #31
    thr_max: float = 0.8               # #31
    final_thr_min: float = 0.15        # F5
    final_thr_max: float = 0.95        # F5
    window_ms: int = 20                # #5 coincidence window
    rest_ms: int = 2                   # D3

    # Lines
    delay_ms: int = 2                  # #11
    min_lines: int = 2                 # R14 / #22
    max_lines: int = 8                 # #21
    fatigue_mult: float = 0.7          # #20 energy × 0.7 per spike
    fatigue_tau_ms: float = 50.0       # #20 back to ~1 within ~200 ms

    # Learning (#17 #18 #19 #30)
    ltp: float = 0.01
    ltd: float = 0.012
    stdp_window_ms: int = 20
    thick_min: float = 0.1
    thick_max: float = 1.0

    # Growth and pruning (#21 #22)
    trial_start: float = 0.15
    trial_permanent: float = 0.3
    trial_ms: int = 500
    unused_ms: int = 10_000
    unused_thin: float = 0.3
    starve_ms: int = 1000
    hebb_gap_ms: int = 3
    hebb_count: int = 5
    hebb_window_ms: int = 1000
    nearest: int = 50                  # F7

    # Homeostasis (#31), run by glia every check (#34)
    target_hz: float = 2.0
    rate_window_ms: int = 5000
    homeo_step: float = 0.001
    homeo_thin: float = 0.99
    check_ms: int = 100

    # Fuels (#29 #30 #36 D5)
    efuel_cost: float = 0.05
    efuel_tau_ms: float = 500.0        # back to ~1 within ~2 s
    cfuel_every_ms: int = 10
    cfuel_rise: float = 0.5
    cfuel_tau_ms: float = 300.0        # fades within ~1 s

    # Locked memory (#35)
    lock_thick: float = 0.8
    lock_ms: int = 60_000
    lock_ltd_scale: float = 0.1

    # Glia (#34)
    glia_group: int = 50
    astro_step: float = 0.05
    rfuel_local_max: float = 2.0

    # Network make-up (#15)
    inhibitory_frac: float = 0.2

    # Reward timing (#30): recent learning stays "eligible" for a reward, fading over ~1 s
    elig_tau_ms: float = 250.0
    # only lines into neurons that fired within this window before the reward are changed (0 = every eligible line)
    reward_window_ms: float = 300.0

    # Neurogenesis (#37, proposed rule from NEXT Q2): a region grows a neuron when 3+ of its neurons are
    # starving while full (8 lines), at most once per 10 s; a neuron with no lines and no spike for 60 s dies
    birth_starving: int = 3
    birth_gap_ms: int = 10_000
    death_ms: int = 60_000
    birth_full_lines: int = 0          # "full" = this many lines (0 = the line limit, v1)

    # ---------- v2 switches (defaults = v1 behaviour; see v2() below) ----------
    leak_tau_ms: float = 0.0           # >0: hub and mini-buckets leak with this time constant, no 20 ms dump
    branches: int = 0                  # >0: mini-buckets per neuron (dendrites) instead of one pool per line
    branch_lo: float = 0.2             # a mini-bucket pours into the hub once it holds this much
    branch_hi: float = 9.0             # ...but overfilled past this it chokes and goes quiet (choke rule)
    dend_gain: float = 1.5             # a pouring mini-bucket adds its charge x this (dendritic spike)
    glia_hz: float = 0.0               # >0: astrocytes calm only neurons above this rate (else above target)
    prune_quiet: int = 1               # 1: quiet + thin lines are pruned (v1); 0: only lines learning made weak
    prune_slices: int = 1              # pruning checks 1/N of the lines each glia check (speed)
    elig_slow_tau_ms: float = 0.0      # >0: second, slow eligibility memory for late rewards
    elig_slow_w: float = 0.0           # its weight in a reward
    protect_ms: float = 0.0            # lines strengthened by a reward are safe from pruning this long
    grow_any: int = 0                  # 1: starving neurons take a line from a neuron that just fired, any layer
    grow_need: float = 0.0             # >0: Hebbian growth only while the neuron fires below this x its target
    reward_grow: int = 0               # research: lines grown per reward toward recent causes (0 = off)
    reward_grow_min: float = 0.2       # ...only for rewards (surprise x gain) at least this big
    reward_grow_ms: float = 300.0      # ...from neurons that fired this recently
    reward_grow_size: float = 0.2      # ...starting strength (a trial line: must reach 0.3 within trial_ms)
    protect_thin: int = 0              # research: rewarded lines are also safe from homeostatic thinning
    close_elig: float = 0.0            # research: eligibility for inputs that bring the hub near threshold


def v2(**over):
    """Zneuron v2: the fix guide's patches and the new features, as one parameter set.

    Reward-guided growth (research 2026-11-02, bees/research.py): v2 delivered 2.9 -> 11.1 squares on basic and
    turned from chance steering to learning; 4 lines per reward beat 8; protecting rewarded lines from thinning and
    'how close to firing' eligibility did not help, so they stay off.
    The brakes were set by activity checks (bee brain, 90 s, no task score used): with no line thinning and glia only
    above 50 Hz the brain ran away (every layer ~38 Hz, lines x3), so each brake is kept in a gentle form instead."""
    p = dict(
        thr_start=0.3, thr_min=0.15, thr_max=0.9,          # fire threshold 0.3 (room to calm busy neurons up to 0.9)
        leak_tau_ms=50.0,                                   # small, slow leak instead of the 20 ms dump
        max_lines=64, birth_full_lines=6,                   # no 8-line limit
        fatigue_mult=0.8, fatigue_tau_ms=30.0,              # softer fatigue
        homeo_thin=0.998, homeo_step=0.001,                 # gentle homeostasis: lines thin -2%/s, not -10%/s
        glia_hz=10.0,                                       # glia calm neurons above 10 Hz
        prune_quiet=0, trial_ms=3000, prune_slices=10,      # prune only weak lines; trials get 3 s; sliced
        branches=4, branch_lo=0.25, branch_hi=0.9, dend_gain=1.5,  # mini-buckets with the choke rule
        protect_ms=300_000.0,                               # lock on reward (5 min)
        grow_any=1, grow_need=2.0,                          # grow by teamwork, only while a neuron needs input
        reward_grow=4,                                      # reward-guided growth: wire up what just caused a reward
        reward_window_ms=0.0,                               # no targeted reward (branch credit does that job)
    )
    p.update(over)
    return Params(**p)
