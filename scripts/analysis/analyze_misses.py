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
CK=Path(sys.argv[1]); seeds=[int(x) for x in sys.argv[2:]]
cfg=OmegaConf.load(CK.parent.parent/"config.yaml")
fab=Fabric(devices=1,accelerator="cpu",num_nodes=1); st=fab.load(str(CK))
cfgh=EnvConfig.from_yaml('configs/target/chase_pursuit_big_v3.yaml'); dt=1.0/cfgh.agent_hz; R=cfgh.target_reach_distance
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
for seed in seeds:
    o,_=env.reset(seed=seed); pl.init_states()
    D=[];T=[];TV=[];done=False;steps=0
    while not done and steps<cfgh.max_episode_steps:
        D.append(raw.backend.get_drone_state().position.copy()); T.append(raw.backend.get_target_state().position.copy())
        TV.append(np.linalg.norm(raw.backend.get_target_state().velocity))
        with torch.no_grad(): a=pl.get_actions(tt(o),greedy=True)
        act=torch.stack(a,-1).cpu().numpy().reshape(env.action_space.shape)
        o,r,term,trunc,info=env.step(act); steps+=1; done=term or trunc
    D=np.array(D);T=np.array(T); dist=np.linalg.norm(D-T,axis=1); TV=np.array(TV)
    within15=int((dist<1.5).sum()); within13=int((dist<1.3).sum())
    tgt_speed=TV.mean()/dt if dt else 0  # approx m/s of target motion... actually TV is body vel
    print(f"seed {seed}: start={dist[0]:.1f}m  min_sep={dist.min():.3f}m (catch<{R})  final={dist[-1]:.1f}m  steps={steps}")
    print(f"   time within 1.5m: {within15} steps ({within15*dt:.0f}s) | within 1.3m: {within13} steps  -> ORBITING just outside")
    print(f"   target mean speed ~{TV.mean():.2f} (units)  | closest approach at step {int(dist.argmin())} ({dist.argmin()*dt:.0f}s)")
