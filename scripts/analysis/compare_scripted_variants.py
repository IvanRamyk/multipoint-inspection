import sys, numpy as np, torch
from pathlib import Path
REPO=Path('/Users/ivan.ramyk/dev/personal/dreamer'); sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/'tools'))
from omegaconf import OmegaConf
from lightning.fabric import Fabric
import gymnasium as gym
from envs.core.config import EnvConfig
from envs.tasks.drone_chase_env import DroneChaseEnv
from envs.sheeprl_wrapper import SheepRLCompatWrapper
from envs.backends.pyflyt_backend import _MAX_SPEED
from sheeprl.algos.dreamer_v3.agent import build_agent
from run_baselines import _TARGET_REL, _TARGET_VEL

CK=Path(sys.argv[1]); N=int(sys.argv[2]) if len(sys.argv)>2 else 50
cfg=OmegaConf.load(CK.parent.parent/"config.yaml")
fab=Fabric(devices=1,accelerator="cpu",num_nodes=1); st=fab.load(str(CK))
cfgh=EnvConfig.from_yaml('configs/target/chase_pursuit_big_v2.yaml'); dt=1.0/cfgh.agent_hz
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
def a_learned(o):
    with torch.no_grad(): a=pl.get_actions(tt(o),greedy=True)
    return torch.stack(a,-1).cpu().numpy().reshape(env.action_space.shape)
def a_pursuit(o,clip=0.7):
    s=o["state"]; rel=s[_TARGET_REL].astype(np.float64); vt=s[_TARGET_VEL].astype(np.float64)
    return np.clip(0.4*(rel+vt),-clip,clip).astype(np.float32)
def a_intercept(o,frac=0.7):
    # fly straight at the predicted INTERCEPT point (constant-bearing collision course)
    s=o["state"]; rel=s[_TARGET_REL].astype(np.float64); vt=s[_TARGET_VEL].astype(np.float64)
    smax=frac*_MAX_SPEED
    a=float(vt@vt)-smax*smax; b=2*float(rel@vt); c=float(rel@rel)
    tau=None
    if abs(a)<1e-6:
        if abs(b)>1e-9: tau=-c/b
    else:
        disc=b*b-4*a*c
        if disc>=0:
            r=np.sqrt(disc); t1=(-b-r)/(2*a); t2=(-b+r)/(2*a)
            cand=[t for t in (t1,t2) if t>1e-3]
            if cand: tau=min(cand)
    aim = rel + vt*tau if tau is not None else rel   # fallback: pure pursuit
    n=np.linalg.norm(aim)
    dirv = aim/n if n>1e-6 else rel
    return np.clip(dirv*frac,-frac,frac).astype(np.float32)

pols={'learned-97%':a_learned,'scripted pure-pursuit(0.7)':lambda o:a_pursuit(o,0.7),'scripted INTERCEPT(0.7)':lambda o:a_intercept(o,0.7)}
res={k:{'caught':[],'steps':{}} for k in pols}
for ep in range(N):
    seed=1000+ep
    for name,fn in pols.items():
        o,_=env.reset(seed=seed)
        if name.startswith('learned'): pl.init_states()
        done=False; steps=0; caught=False
        while not done and steps<cfgh.max_episode_steps:
            o,r,term,trunc,info=env.step(fn(o)); steps+=1; done=term or trunc; caught=bool(info.get("is_success",False))
        res[name]['caught'].append(caught)
        if caught: res[name]['steps'][seed]=steps
common=set.intersection(*[set(res[k]['steps'].keys()) for k in pols])
print(f"\n=== {N} eps, realistic 1.2m, same seeds; {len(common)} caught by all ===")
print(f"{'policy':28} {'success':>8} {'catch time own':>15} {'catch time common':>18}")
for k in pols:
    sr=np.mean(res[k]['caught']); own=np.mean([res[k]['steps'][s] for s in res[k]['steps']])*dt
    com=np.mean([res[k]['steps'][s] for s in common])*dt if common else float('nan')
    print(f"{k:28} {sr*100:6.0f}%  {own:11.1f}s  {com:14.1f}s")
