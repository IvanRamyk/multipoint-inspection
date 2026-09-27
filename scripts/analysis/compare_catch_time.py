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
from run_baselines import pursuit_action, _TARGET_REL, _TARGET_VEL

CK=Path(sys.argv[1]); N=int(sys.argv[2]) if len(sys.argv)>2 else 60
cfg=OmegaConf.load(CK.parent.parent/"config.yaml")
fab=Fabric(devices=1,accelerator="cpu",num_nodes=1); st=fab.load(str(CK))
cfgh=EnvConfig.from_yaml('configs/target/chase_pursuit_big_v2.yaml')
raw=DroneChaseEnv(config=cfgh); env=SheepRLCompatWrapper(raw)
dt=1.0/cfgh.agent_hz
sc=cfg.env.screen_size; ch=1 if cfg.env.grayscale else 3
obs_space=gym.spaces.Dict({"depth":gym.spaces.Box(0,255,(ch,sc,sc),dtype=np.uint8),"state":env.observation_space["state"]})
wm,_,_,_,pl=build_agent(fab,[env.action_space.shape[0]],True,cfg,obs_space,world_model_state=st["world_model"],actor_state=st["actor"])
pl.num_envs=1; pl.eval(); cnn=list(cfg.algo.cnn_keys.encoder)
def tt(o):
    out={}
    for k,v in o.items():
        t=torch.from_numpy(v.copy()).float()
        out[k]=(t.permute(2,0,1).unsqueeze(0).unsqueeze(0)/255.0-0.5) if k in cnn else t.view(1,1,-1)
    return out
def learned(o):
    with torch.no_grad(): a=pl.get_actions(tt(o),greedy=True)
    return torch.stack(a,-1).cpu().numpy().reshape(env.action_space.shape)
def scripted(clip):
    def f(o):
        s=o["state"]; rel=s[_TARGET_REL].astype(np.float64); vel=s[_TARGET_VEL].astype(np.float64)
        return np.clip(0.4*(rel+1.0*vel),-clip,clip).astype(np.float32)
    return f

policies={'learned-97%':('nn',None),'scripted-0.5(default)':('sc',0.5),'scripted-0.7(tuned)':('sc',0.7)}
res={k:{'caught':[],'steps':{}} for k in policies}
for ep in range(N):
    seed=1000+ep
    for name,(kind,clip) in policies.items():
        o,_=env.reset(seed=seed)
        if kind=='nn': pl.init_states()
        done=False; steps=0; caught=False
        while not done and steps<cfgh.max_episode_steps:
            act=learned(o) if kind=='nn' else scripted(clip)(o)
            o,r,term,trunc,info=env.step(act); steps+=1; done=term or trunc
            caught=bool(info.get("is_success",False))
        res[name]['caught'].append(caught)
        if caught: res[name]['steps'][seed]=steps
print(f"\n=== {N} episodes, realistic catch 1.2m, same seeds ===")
common=set.intersection(*[set(res[k]['steps'].keys()) for k in policies])
print(f"episodes caught by ALL three: {len(common)}\n")
print(f"{'policy':24} {'success':>8} {'avg catch time (own)':>22} {'avg catch time (common)':>24}")
for k in policies:
    sr=np.mean(res[k]['caught'])
    own=np.mean([res[k]['steps'][s] for s in res[k]['steps']])*dt
    com=np.mean([res[k]['steps'][s] for s in common])*dt if common else float('nan')
    print(f"{k:24} {sr*100:6.0f}%  {own:14.1f}s ({own/dt:.0f} st) {com:16.1f}s ({com/dt:.0f} st)")
