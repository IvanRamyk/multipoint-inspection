import sys, numpy as np, torch
from pathlib import Path
REPO=Path('/Users/ivan.ramyk/dev/personal/dreamer'); sys.path.insert(0,str(REPO))
from omegaconf import OmegaConf
from lightning.fabric import Fabric
import gymnasium as gym
from envs.core.config import EnvConfig
from envs.tasks.drone_chase_env import DroneChaseEnv
from envs.sheeprl_wrapper import SheepRLCompatWrapper
from sheeprl.algos.dreamer_v3.agent import build_agent
ck=Path(sys.argv[1]); cfg=OmegaConf.load(ck.parent.parent/"config.yaml")
fab=Fabric(devices=1,accelerator="cpu",num_nodes=1); st=fab.load(str(ck))
cfgh=EnvConfig.from_yaml('configs/target/chase_pursuit_big.yaml')
raw=DroneChaseEnv(config=cfgh); env=SheepRLCompatWrapper(raw)
sc=cfg.env.screen_size; ch=1 if cfg.env.grayscale else 3
obs_space=gym.spaces.Dict({"depth":gym.spaces.Box(0,255,(ch,sc,sc),dtype=np.uint8),"state":env.observation_space["state"]})
wm,_,_,_,pl=build_agent(fab,[env.action_space.shape[0]],True,cfg,obs_space,world_model_state=st["world_model"],actor_state=st["actor"])
pl.num_envs=1; pl.eval(); wm.eval(); cnn=list(cfg.algo.cnn_keys.encoder)
def tt(o):
    out={}
    for k,v in o.items():
        t=torch.from_numpy(v.copy()).float()
        out[k]=(t.permute(2,0,1).unsqueeze(0).unsqueeze(0)/255.0-0.5) if k in cnn else t.view(1,1,-1)
    return out
res=[]
for seed in range(1000,1025):
    o,_=env.reset(seed=seed); pl.init_states()
    for mid in list(getattr(raw.backend,"_target_marker_ids",[]) or []):
        try: raw.backend._aviary.removeBody(mid)
        except: pass
    done=False; steps=0; caught=False
    while not done and steps<1000:
        with torch.no_grad(): a=pl.get_actions(tt(o),greedy=True)
        act=torch.stack(a,-1).cpu().numpy().reshape(env.action_space.shape)
        o,r,term,trunc,info=env.step(act); done=term or trunc; steps+=1
        caught=bool(info.get("is_success",False))
    res.append((seed,caught,steps))
    print(f"seed {seed}: caught={caught} steps={steps}", flush=True)
c=[r for r in res if r[1]]
print(f"\nRENDER catch rate: {len(c)}/{len(res)}")
print("good catches (moderate len):", [r for r in c if 150<r[2]<450][:6])
