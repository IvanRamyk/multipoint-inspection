import sys, numpy as np, torch
from pathlib import Path
REPO=Path('/Users/ivan.ramyk/dev/personal/dreamer'); sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/'tools'))
from envs.core.config import EnvConfig
from envs.tasks.drone_chase_env import DroneChaseEnv
from envs.sheeprl_wrapper import SheepRLCompatWrapper
from envs.backends.pyflyt_backend import _MAX_SPEED
from run_baselines import _TARGET_REL, _TARGET_VEL
cfgh=EnvConfig.from_yaml('configs/target/chase_pursuit_big_v2.yaml'); dt=1.0/cfgh.agent_hz
raw=DroneChaseEnv(config=cfgh); env=SheepRLCompatWrapper(raw)
def solve_tau(rel,vt,smax):
    a=float(vt@vt)-smax*smax; b=2*float(rel@vt); c=float(rel@rel)
    if abs(a)<1e-6: return (-c/b) if abs(b)>1e-9 and -c/b>1e-3 else None
    disc=b*b-4*a*c
    if disc<0: return None
    r=np.sqrt(disc); cand=[t for t in ((-b-r)/(2*a),(-b+r)/(2*a)) if t>1e-3]
    return min(cand) if cand else None
def pursuit(rel,vt,clip): return np.clip(0.4*(rel+vt),-clip,clip)
def intercept(rel,vt,frac):
    tau=solve_tau(rel,vt,frac*_MAX_SPEED); aim=rel+vt*tau if tau is not None else rel
    n=np.linalg.norm(aim); d=aim/n if n>1e-6 else rel; return np.clip(d*frac,-frac,frac)
def hybrid(rel,vt,frac,switch):
    dist=np.linalg.norm(rel)
    if dist>switch: return intercept(rel,vt,frac)
    return pursuit(rel,vt,frac)   # pure pursuit for stable final approach
def partial(rel,vt,frac,alpha):  # blended lead: aim at rel + alpha*vt*tau
    tau=solve_tau(rel,vt,frac*_MAX_SPEED); aim=rel+(alpha*vt*tau if tau is not None else vt)
    n=np.linalg.norm(aim); d=aim/n if n>1e-6 else rel; return np.clip(d*frac,-frac,frac)

def act(name,o):
    s=o["state"]; rel=s[_TARGET_REL].astype(np.float64); vt=s[_TARGET_VEL].astype(np.float64)
    if name=='pursuit': return pursuit(rel,vt,0.7).astype(np.float32)
    if name=='intercept': return intercept(rel,vt,0.7).astype(np.float32)
    if name=='hybrid(switch 5m)': return hybrid(rel,vt,0.7,5.0).astype(np.float32)
    if name=='hybrid(switch 8m)': return hybrid(rel,vt,0.7,8.0).astype(np.float32)
    if name=='partial-lead(0.5)': return partial(rel,vt,0.7,0.5).astype(np.float32)
N=int(sys.argv[1]) if len(sys.argv)>1 else 50
pols=['pursuit','intercept','hybrid(switch 5m)','hybrid(switch 8m)','partial-lead(0.5)']
res={k:{'c':[],'st':{}} for k in pols}
for ep in range(N):
    seed=1000+ep
    for name in pols:
        o,_=env.reset(seed=seed); done=False; steps=0; caught=False
        while not done and steps<cfgh.max_episode_steps:
            o,r,term,trunc,info=env.step(act(name,o)); steps+=1; done=term or trunc; caught=bool(info.get("is_success",False))
        res[name]['c'].append(caught)
        if caught: res[name]['st'][seed]=steps
common=set.intersection(*[set(res[k]['st'].keys()) for k in pols])
print(f"\n=== {N} eps realistic 1.2m; {len(common)} caught by ALL ===")
print(f"{'controller':22} {'success':>8} {'time own':>10} {'time common':>13}")
for k in pols:
    sr=np.mean(res[k]['c']); own=np.mean([res[k]['st'][s] for s in res[k]['st']])*dt if res[k]['st'] else float('nan')
    com=np.mean([res[k]['st'][s] for s in common])*dt if common else float('nan')
    print(f"{k:22} {sr*100:6.0f}%  {own:8.1f}s  {com:11.1f}s")
