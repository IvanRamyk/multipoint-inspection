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
CK=Path(sys.argv[1]); seed=int(sys.argv[2]); out=sys.argv[3]
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
o,_=env.reset(seed=seed); pl.init_states()
D=[];T=[];done=False;steps=0;c=False
while not done and steps<cfgh.max_episode_steps:
    D.append(raw.backend.get_drone_state().position.copy()); T.append(raw.backend.get_target_state().position.copy())
    with torch.no_grad(): a=pl.get_actions(tt(o),greedy=True)
    act=torch.stack(a,-1).cpu().numpy().reshape(env.action_space.shape)
    o,r,term,trunc,info=env.step(act); steps+=1; done=term or trunc; c=bool(info.get("is_success",False))
D=np.array(D);T=np.array(T); dist=np.linalg.norm(D-T,axis=1)
np.savez(out, drone=D, target=T, caught=c, reach_distance=cfgh.target_reach_distance, dome_size=cfgh.flight_dome_size, dt=dt)
print(f"seed {seed}: caught={c} steps={steps} start={dist[0]:.1f}m min_sep={dist.min():.2f}m (catch<{cfgh.target_reach_distance}) final={dist[-1]:.1f}m")
