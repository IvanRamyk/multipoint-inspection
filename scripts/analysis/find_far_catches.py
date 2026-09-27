import sys, numpy as np, torch, json
from pathlib import Path
REPO=Path('/Users/ivan.ramyk/dev/personal/dreamer'); sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/'tools'))
from omegaconf import OmegaConf
from lightning.fabric import Fabric
import gymnasium as gym
from envs.core.config import EnvConfig
from envs.tasks.drone_chase_env import DroneChaseEnv
from envs.sheeprl_wrapper import SheepRLCompatWrapper
from sheeprl.algos.dreamer_v3.agent import build_agent
CK=Path(sys.argv[1]); N=int(sys.argv[2]); base=int(sys.argv[3])
OUT=Path('results/chase_big_report/far_catches'); OUT.mkdir(parents=True,exist_ok=True)
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
# pass 1: find start distances cheaply (reset only)
starts=[]
for i in range(N):
    seed=base+i; env.reset(seed=seed)
    d=float(np.linalg.norm(raw.backend.get_drone_state().position-raw.backend.get_target_state().position))
    starts.append((seed,d))
starts.sort(key=lambda x:-x[1])
cands=starts[:14]
print("farthest starts:", [(s,round(d,1)) for s,d in cands[:8]], flush=True)
saved=[]
for seed,sd in cands:
    o,_=env.reset(seed=seed); pl.init_states()
    D=[];T=[];done=False;steps=0;caught=False
    while not done and steps<MAX:
        D.append(raw.backend.get_drone_state().position.copy()); T.append(raw.backend.get_target_state().position.copy())
        with torch.no_grad(): a=pl.get_actions(tt(o),greedy=True)
        act=torch.stack(a,-1).cpu().numpy().reshape(env.action_space.shape)
        o,r,term,trunc,info=env.step(act); steps+=1; done=term or trunc; caught=bool(info.get("is_success",False))
    print(f"seed {seed}: start={sd:.1f}m caught={caught} steps={steps} ({steps*dt:.1f}s)", flush=True)
    if caught:
        D=np.array(D);T=np.array(T)
        f=OUT/f"farcatch_seed{seed}_start{sd:.0f}m_{steps*dt:.0f}s.npz"
        np.savez(f, drone=D, target=T, caught=True, reach_distance=cfgh.target_reach_distance, dome_size=cfgh.flight_dome_size, dt=dt)
        saved.append((seed,round(sd,1),steps,str(f)))
        if len(saved)>=4: break
json.dump(saved, open(OUT/'index.json','w')); print("SAVED",saved); print("FAR_DONE")
