import sys, numpy as np, torch
from pathlib import Path
REPO=Path('/Users/ivan.ramyk/dev/personal/dreamer'); sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/'tools'))
from omegaconf import OmegaConf
from lightning.fabric import Fabric
import gymnasium as gym
from envs.core.config import EnvConfig
from envs.tasks.drone_chase_env import DroneChaseEnv
from envs.sheeprl_wrapper import SheepRLCompatWrapper
from sheeprl.algos.dreamer_v3.agent import build_agent
CK=Path(sys.argv[1]); N=int(sys.argv[2]) if len(sys.argv)>2 else 100; base=int(sys.argv[3]) if len(sys.argv)>3 else 1000
cfg=OmegaConf.load(CK.parent.parent/"config.yaml")
fab=Fabric(devices=1,accelerator="cpu",num_nodes=1); st=fab.load(str(CK))
cfgh=EnvConfig.from_yaml('configs/target/chase_pursuit_big_v2_time.yaml'); dt=1.0/cfgh.agent_hz
raw=DroneChaseEnv(config=cfgh); env=SheepRLCompatWrapper(raw)
scr=cfg.env.screen_size; ch=1 if cfg.env.grayscale else 3
obs_space=gym.spaces.Dict({"depth":gym.spaces.Box(0,255,(ch,scr,scr),dtype=np.uint8),"state":env.observation_space["state"]})
wm,_,_,_,pl=build_agent(fab,[env.action_space.shape[0]],True,cfg,obs_space,world_model_state=st["world_model"],actor_state=st["actor"])
pl.num_envs=1; pl.eval(); cnn=list(cfg.algo.cnn_keys.encoder)
def tt(o):
    out={}
    for k,v in o.items():
        t=torch.from_numpy(v.copy()).float(); out[k]=(t.permute(2,0,1).unsqueeze(0).unsqueeze(0)/255.0-0.5) if k in cnn else t.view(1,1,-1)
    return out
caught=0; miss_genuine=0; miss_targetcrash=0; details=[]
for ep in range(N):
    seed=base+ep; o,_=env.reset(seed=seed); pl.init_states()
    done=False; steps=0; c=False; Tz=[]; Tpos=[]; minsep=1e9
    while not done and steps<cfgh.max_episode_steps:
        with torch.no_grad(): a=pl.get_actions(tt(o),greedy=True)
        act=torch.stack(a,-1).cpu().numpy().reshape(env.action_space.shape)
        tp=raw.backend.get_target_state().position; Tz.append(float(tp[2])); Tpos.append(tp.copy())
        o,r,term,trunc,info=env.step(act); steps+=1; done=term or trunc; c=bool(info.get("is_success",False))
    if c: caught+=1; continue
    Tz=np.array(Tz); Tpos=np.array(Tpos)
    # target-crash heuristics: target hit ground, or got stuck (barely moved in last 200 steps)
    hit_ground = Tz.min()<0.5
    stuck = len(Tpos)>200 and np.linalg.norm(Tpos[-1]-Tpos[-200])<1.0
    if hit_ground or stuck:
        miss_targetcrash+=1; details.append((seed,'TARGET_CRASH' if hit_ground else 'TARGET_STUCK', round(Tz.min(),2)))
    else:
        miss_genuine+=1; details.append((seed,'genuine_miss',round(Tz.min(),2)))
tot=N
print(f"\n=== @{CK.name} , {N} eps (seed base {base}), max_steps={cfgh.max_episode_steps} ({cfgh.max_episode_steps*dt:.0f}s) ===")
print(f"caught: {caught}/{tot} = {100*caught/tot:.0f}% (raw)")
print(f"misses: {miss_genuine} genuine + {miss_targetcrash} target-crash/stuck")
fair_den=tot-miss_targetcrash
print(f"FAIR success (excluding target self-crashes): {caught}/{fair_den} = {100*caught/fair_den:.1f}%")
for d in details: print('  miss', d)
