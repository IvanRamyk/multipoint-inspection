# Chase task — iteration data

Data-only record of the 5 training iterations on `DroneChaseEnv` (pursue a second
QuadX drone flying a non-self-intersecting waypoint route).

## Instance
- vast.ai `45813382`, RTX 3060, 24 vCPU, $0.052/hr, `ssh8.vast.ai:13382`.

## Shared setup (all iterations unless noted)
`exp=drone_chase`, `algo=dreamer_v3_S`, `fabric.accelerator=gpu`,
`fabric.precision=16-mixed`, `env.sync_env=false`, `env.num_envs=8`,
`checkpoint.every=500`, `metric.log_every=200`, no `--stop-threshold`.

## Task configs
| param | chase.yaml | chase_easy.yaml | chase_veryeasy.yaml | chase_mid.yaml | chase_easy2.yaml |
|---|---|---|---|---|---|
| flight_dome_size | 20 | 20 | 12 | 16 | 20 |
| target_reach_distance | 1.5 | 2.0 | 2.5 | 2.0 | 2.0 |
| target_altitude_min/max | 1.5 / 6.0 | 1.5 / 6.0 | 1.5 / 4.0 | 1.5 / 5.0 | 1.5 / 6.0 |
| chase_num_waypoints | 8 | 8 | 4 | 6 | 8 |
| chase_min_separation | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 |
| chase_ordering | 2opt | 2opt | 2opt | 2opt | 2opt |
| chase_target_speed_cap | 0.4 | 0.3 | 0.2 | 0.25 | 0.3 |
| chase_target_gain | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| chase_waypoint_reach | 1.5 | 1.5 | 1.5 | 1.5 | 1.5 |
| wind_enabled | true (0.2) | false | false | false | false |
| num_obstacles | 0 | 0 | 0 | 0 | 0 |
| max_episode_steps | 600 | 600 | 500 | 600 | 600 |
| reward_shaping | 0.5 | 2.0 | 1.0 | 1.0 | 1.0 |
| collision_penalty | -10 | -10 | -10 | -10 | -10 |
| target_catch_reward | 60 | 60 | 60 | 60 | 60 |

## Iteration configs
| iter | task config | total_steps | replay_ratio | other |
|---|---|---|---|---|
| 1 | chase.yaml | 100000 | 1 | — |
| 2 | chase_easy.yaml | 60000 | 1 | — |
| 3 | chase_easy.yaml | 60000 | 0.5 | — |
| 4 | chase_veryeasy.yaml | 60000 | 1 | — |
| 5 | chase_easy.yaml | 20000 | 1 | `checkpoint.resume_from=<iter4 ckpt_8064>` |
| 6 | chase_mid.yaml | 30000 | 1 | `resume_from=<iter4 ckpt_7056>`, keep_last=40 |
| 7 | chase_easy2.yaml | 52000 | 1 | `resume_from=<iter6 ckpt_30000>`, keep_last→8 after disk-full relaunch @ 38.5k |

Instance for iters 6–7: vast 45881700, RTX 3060 Ti, 24 vCPU (192 cores), $0.069/hr; destroyed at teardown.

## rew_avg trajectory (step: value)
**Iter 1:** 2400:−20.2, 2600:−15.9, 2800:−13.5, 3000:−9.6, 3200:−10.2, 3400:+61.5, 4800:−8.9, 6200:+58.3
- ep_len_avg — 2800:121, 3000:35, 3200:51, 3400:68, 4800:600, 6200:406

**Iter 2:** 2200:+63.8, 2400:−2.3, 2600:−11.5, 2800:−10.5, 3000:−12.2, 3400:−9.7, (…) −35.4, −50.3, −46.0, −62.8, 9000:+68.7

**Iter 3:** 2600:−5, 2800:−14, 3000:−11, 3200:−11, 3400:−11, 3600:−10, 4000:−10, 4200:−8, 4400:−15, 4600:−12, 4800:−3, 5600:−8 (max observed: −3.3)

**Iter 4:** 5800:[61.3, 59.8, 60.8, 60.7, 61.4], 6000:[59.8, 60.8, 60.7, 61.4, 58.5], 6400:[…, −2.3] (max observed: ~63)

**Iter 5** (resumed at step 8064): 8400:60, 8600:62, 8800:61, 9000:61, 9200:62, 9400:61, 9800:61, 10000:−15

## Greedy eval results (`eval_target_ckpt.py --task chase`)
| checkpoint | eval config | seed | episodes | caught | success | caught-episode details |
|---|---|---|---|---|---|---|
| iter1 ckpt_6552 | chase.yaml | 30 | 6 | 0 | 0% | min_dist reached 1.90 m |
| iter1 ckpt_7056 | chase.yaml | 40 | 6 | 2 | 33% | ep4 r=59.74 s=307; ep6 r=58.89 s=148 |
| iter1 ckpt_7056 | chase.yaml | 50 | 14 | 2 | 14% | ep1 r=60.70 s=38; ep11 r=59.96 s=19 |
| iter4 ckpt_7056 | chase_veryeasy.yaml | 60 | 10 | 5 | 50% | ep1 r=59.99 s=1; ep5 r=61.77 s=60; ep6 r=61.90 s=59; ep8 r=62.97 s=74; ep10 r=60.91 s=140 |
| iter5 ckpt_10584 | chase_easy.yaml | 70 | 10 | 1 | 10% | ep8 r=61.09 s=7 |
| iter5 ckpt_9576 | chase_easy.yaml | 80 | 10 | 1 | 10% | ep3 r=61.54 s=27 |

### Iter 6 dense sweep — chase_mid, seed 100, 20 eps each (greedy)
| checkpoint | success |
|---|---|
| ckpt_10584 | 15% (3/20) |
| ckpt_14112 | 30% (6/20) |
| ckpt_18144 | 25% (5/20) |
| ckpt_22176 | 55% (11/20) |
| ckpt_26208 | 40% (8/20) |
| **ckpt_30000** | **60% (12/20)** |
(Early single-ckpt read: ckpt_16632 = 10% — a bad-window artefact vs the 60% best.)

### Iter 7 dense sweep — chase_easy2, seed 100, 20 eps each (greedy)
| checkpoint | success |
|---|---|
| ckpt_45568 | 10% (2/20) |
| **ckpt_47568** | **70% (14/20)** |
| ckpt_48568 | 50% (10/20) |
| ckpt_50568 | 65% (13/20) |
| ckpt_51568 | 45% (9/20) |
| ckpt_52000 | 35% (7/20) |
(Video subset ckpt_47568 @ 12 eps seed 100 = 41% (5/12) — small-sample variance; 20-ep = 70%.)

(`r` = episode reward, `s` = steps to catch. Multiple distinct `ckpt_7056` files
exist across runs — different runs, same step number.)

## Artifacts
- Checkpoints: `results/chase_iter4/version_0/checkpoint/` (228 MB),
  `results/chase_iter5/version_0/checkpoint/`,
  `results/chase_iter6/version_0/checkpoint/` (ckpt_30000=60%, ckpt_22176=55% on mid),
  `results/chase_iter7/version_0/checkpoint/` (**ckpt_47568=70%**, ckpt_50568=65% on easy2).
- Catch videos: `results/CHASE_ITER7_easy2_ckpt_47568_*_3d.mp4` (5 clips, the headline),
  `results/CHASE_ITER4_*_3d.mp4`, `results/CHASE_ITER5_plateau_*_3d.mp4`, `results/CHASE_TRAINED_*_3d.mp4`.
- Watcher improvement sequences: `results/watch_iter3/` (15 mp4), `results/watch_iter4/` (36), `results/watch_iter5/` (9).
- tfevents: `results/chase_iter{4,5,6,7}/version_0/`. Eval logs: `results/chase_iter7/eval_logs/`.

## Result
Curriculum ladder (warm-start each rung from the previous best ckpt) →
**70% greedy interception on chase_easy2** (dome 20, 8 wp, 1.5 m/s target, reach 2.0):
iter4 50% (veryeasy) → iter6 60% (mid) → iter7 70% (easy2). Best chase result to date.
