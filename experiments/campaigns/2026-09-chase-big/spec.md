# Campaign: chase-big — realistic 45 cm drone, state-only, fast throughput

## Why
The chase-pursuit (cf2x nano) line established that reward+geometry shaping produces active
but imprecise, undertrained pursuit that fails the alignment gate within budget — the binding
constraint was **throughput** (~1.5 sps, image-based DreamerV3 on weak GPUs). This campaign
rebuilds the task to be realistic and fast:

- **Airframe:** `primitive_drone` (~45 cm motor-to-motor, 1 kg) instead of the 9 cm cf2x.
- **Observation:** state-only (`observe_depth: false`) — the target's relative pos+vel is in the
  state vector, so the depth camera carried no task info. Dropping it cuts the world model
  7.94M→2.44M params (−69%) and removes the per-step camera render (−29%).
- **Speeds (achievable envelope):** pursuer ~5 m/s cap (achieves ~3 chasing), target 2.0 m/s
  (cap 0.40). Scripted baseline **suitable: 70% success, alignment 0.80, random 0%**.
- **Geometry:** ~38 m mean spawn, 1200-step (40 s) episodes, catch radius 1.2 m.
- **Reward:** catch 150 flat + up to 150 time-decay bonus (immediate=300, late=150) so catching
  ASAP is strictly optimal; collision −1000 anti-crash; telescoping shaping 1.0; distance_penalty 0.01.

Config: `configs/target/chase_pursuit_big.yaml`.

## PROBE — throughput + GPU utilisation + model-size ablation (this step)
Before committing to a long run, empirically establish:
1. **sps speedup** vs the old ~1.5 sps image baseline (state-only + more envs).
2. **GPU utilisation** (qualitative, `nvidia-smi` on the box) — with a 2.4–4.4M param model a
   powerful GPU may be under-utilised; measure before paying for more.
3. **XS vs S** (dreamer_v3_XS vs _S): which converts wall-clock into pursuit better on this
   tiny 15-dim task. state-only S (4.41M) is still lighter than the old image XS (7.94M).

Runs launched on powerful GPUs via `tools/launch_hard.py` (sorts offers by dlperf). num_envs 8.
Exp configs: `drone_chase_fast` (XS), `drone_chase_fast_S` (S). This is a probe, not a gated rung;
acceptance criteria for the full run are set after the probe informs size + GPU choice.

## Ground rules
- Budget guard: $15 cap, fresh TTL. Concurrency for the probe ≤ 4.
- Speeds/geometry are the achievable-envelope choice (real 25 m/s needs a flight-controller
  retune — deferred). Scripted baseline must stay `suitable` (it is: 70%).
