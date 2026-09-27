import sys, json, numpy as np, torch, time
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
CK=Path(sys.argv[1]); N=int(sys.argv[2]) if len(sys.argv)>2 else 1000; base=int(sys.argv[3]) if len(sys.argv)>3 else 1000
cfg=OmegaConf.load(CK.parent.parent/"config.yaml")
fab=Fabric(devices=1,accelerator="cpu",num_nodes=1); st=fab.load(str(CK))
cfgh=EnvConfig.from_yaml('configs/target/chase_pursuit_big_v3.yaml'); dt=1.0/cfgh.agent_hz; MAX=cfgh.max_episode_steps
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
def run(seed,kind):
    o,_=env.reset(seed=seed)
    if kind=='nn': pl.init_states()
    done=False;steps=0;caught=False;msep=1e9
    while not done and steps<MAX:
        if kind=='nn':
            with torch.no_grad(): a=pl.get_actions(tt(o),greedy=True)
            act=torch.stack(a,-1).cpu().numpy().reshape(env.action_space.shape)
        else:
            s=o["state"]; act=np.clip(0.4*(s[_TARGET_REL].astype(np.float64)+s[_TARGET_VEL].astype(np.float64)),-0.7,0.7).astype(np.float32)
        d=np.linalg.norm(raw.backend.get_drone_state().position-raw.backend.get_target_state().position); msep=min(msep,d)
        o,r,term,trunc,info=env.step(act); steps+=1; done=term or trunc; caught=bool(info.get("is_success",False))
    cat='CAUGHT' if caught else ('CRASH' if steps<MAX else ('NEARMISS' if msep<2.0 else 'WANDER'))
    return caught,steps,cat
R={'nn':{'c':0,'steps':[],'cat':{}},'sc':{'c':0,'steps':[],'cat':{}}}
common=[]; t0=time.time()
for i in range(N):
    seed=base+i
    lc,ls,lcat=run(seed,'nn'); sc,ss,scat=run(seed,'sc')
    for who,(c,s,cat) in [('nn',(lc,ls,lcat)),('sc',(sc,ss,scat))]:
        if c: R[who]['c']+=1; R[who]['steps'].append(s)
        R[who]['cat'][cat]=R[who]['cat'].get(cat,0)+1
    if lc and sc: common.append((ls,ss))
    if (i+1)%50==0:
        el=time.time()-t0
        print(f"[{i+1}/{N}] {el/60:.0f}min  learned {R['nn']['c']}/{i+1}={100*R['nn']['c']/(i+1):.1f}%  scripted {R['sc']['c']}/{i+1}={100*R['sc']['c']/(i+1):.1f}%",flush=True)
        json.dump({'done':i+1,'learned':R['nn'],'scripted':R['sc']},open('/tmp/h2h_partial.json','w'))
print("\n===== FINAL 1000-EP HEAD-TO-HEAD (realistic 1.2m, same start positions) =====")
for who,name in [('nn','LEARNED (DreamerV3 @332k)'),('sc','SCRIPTED (tuned pursuit 0.7)')]:
    c=R[who]['c']; ct=np.mean(R[who]['steps'])*dt if R[who]['steps'] else 0
    print(f"{name:32} success {c}/{N} = {100*c/N:.1f}%   avg catch {ct:.1f}s   breakdown {R[who]['cat']}")
if common:
    ln=np.mean([x[0] for x in common])*dt; sn=np.mean([x[1] for x in common])*dt
    print(f"catch time on {len(common)} common episodes: learned {ln:.1f}s vs scripted {sn:.1f}s")
json.dump({'learned':R['nn'],'scripted':R['sc'],'N':N},open('results/chase_big_report/head2head_1000.json','w'))
print("DONE")
