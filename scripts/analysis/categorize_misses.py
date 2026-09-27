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
CK=Path(sys.argv[1]); N=int(sys.argv[2]); base=int(sys.argv[3]) if len(sys.argv)>3 else 1000
cfg=OmegaConf.load(CK.parent.parent/"config.yaml")
fab=Fabric(devices=1,accelerator="cpu",num_nodes=1); st=fab.load(str(CK))
cfgh=EnvConfig.from_yaml('configs/target/chase_pursuit_big_v2_time.yaml'); dt=1.0/cfgh.agent_hz; MAX=cfgh.max_episode_steps
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
cats={'CAUGHT':0,'PURSUER_CRASH':0,'NEAR_MISS(<2m)':0,'WANDER':0,'TARGET_PROBLEM':0}; misslog=[]
for ep in range(N):
    seed=base+ep; o,_=env.reset(seed=seed); pl.init_states()
    D=[];T=[];done=False;steps=0;c=False
    while not done and steps<MAX:
        D.append(raw.backend.get_drone_state().position.copy()); T.append(raw.backend.get_target_state().position.copy())
        with torch.no_grad(): a=pl.get_actions(tt(o),greedy=True)
        act=torch.stack(a,-1).cpu().numpy().reshape(env.action_space.shape)
        o,r,term,trunc,info=env.step(act); steps+=1; done=term or trunc; c=bool(info.get("is_success",False))
    if c: cats['CAUGHT']+=1; continue
    D=np.array(D);T=np.array(T); dist=np.linalg.norm(D-T,axis=1); msep=dist.min()
    tgt_min_alt=T[:,2].min(); tgt_stuck=len(T)>200 and np.linalg.norm(T[-1]-T[-200])<1.0
    ended_early = steps<MAX
    if tgt_min_alt<0.5 or tgt_stuck: cat='TARGET_PROBLEM'
    elif ended_early: cat='PURSUER_CRASH'
    elif msep<2.0: cat='NEAR_MISS(<2m)'
    else: cat='WANDER'
    cats[cat]+=1; misslog.append((seed,cat,round(msep,2),steps))
tot=N; caught=cats['CAUGHT']; tp=cats['TARGET_PROBLEM']
print(f"\n=== @{CK.name}, {N} eps (base {base}), {MAX} steps ({MAX*dt:.0f}s) ===")
for k,v in cats.items(): print(f"  {k:18}: {v}  ({100*v/tot:.1f}%)")
print(f"raw success: {100*caught/tot:.1f}%")
print(f"FAIR success (exclude target problems): {caught}/{tot-tp} = {100*caught/(tot-tp):.1f}%")
print("misses:", misslog)
