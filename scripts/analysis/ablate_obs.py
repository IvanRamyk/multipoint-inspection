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
CK=Path(sys.argv[1]); N=int(sys.argv[2]); MODE=sys.argv[3]  # none | zero_xy | shift_xy | zero_z
cfg=OmegaConf.load(CK.parent.parent/"config.yaml")
fab=Fabric(devices=1,accelerator="cpu",num_nodes=1); st=fab.load(str(CK))
cfgh=EnvConfig.from_yaml('configs/target/chase_pursuit_big_v3.yaml'); MAX=cfgh.max_episode_steps
raw=DroneChaseEnv(config=cfgh); env=SheepRLCompatWrapper(raw)
scr=cfg.env.screen_size; ch=1 if cfg.env.grayscale else 3
obs_space=gym.spaces.Dict({"depth":gym.spaces.Box(0,255,(ch,scr,scr),dtype=np.uint8),"state":env.observation_space["state"]})
wm,_,_,_,pl=build_agent(fab,[env.action_space.shape[0]],True,cfg,obs_space,world_model_state=st["world_model"],actor_state=st["actor"])
pl.num_envs=1; pl.eval(); cnn=list(cfg.algo.cnn_keys.encoder)
def perturb(o):
    s=o["state"].copy()
    if MODE=="zero_xy": s[0]=0.0; s[1]=0.0            # прибрати абсолютні x,y
    elif MODE=="shift_xy": s[0]+=200.0; s[1]-=200.0   # зсунути «де я на мапі»
    elif MODE=="zero_z": s[2]=0.0                     # прибрати висоту
    o=dict(o); o["state"]=s; return o
def tt(o):
    out={}
    for k,v in o.items():
        t=torch.from_numpy(v.copy()).float(); out[k]=(t.permute(2,0,1).unsqueeze(0).unsqueeze(0)/255.0-0.5) if k in cnn else t.view(1,1,-1)
    return out
caught=0; crash=0
for i in range(N):
    o,_=env.reset(seed=1000+i); pl.init_states()
    done=False;steps=0;c=False
    while not done and steps<MAX:
        with torch.no_grad(): a=pl.get_actions(tt(perturb(o)),greedy=True)
        act=torch.stack(a,-1).cpu().numpy().reshape(env.action_space.shape)
        o,r,term,trunc,info=env.step(act); steps+=1; done=term or trunc; c=bool(info.get("is_success",False))
    caught+=c
    if not c and steps<MAX: crash+=1
print(f"MODE={MODE:10} success {caught}/{N} = {100*caught/N:.0f}%   crashes={crash}")
