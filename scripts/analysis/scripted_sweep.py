import sys, numpy as np
from pathlib import Path
REPO=Path('/Users/ivan.ramyk/dev/personal/dreamer'); sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/'tools'))
from envs.core.config import EnvConfig
from envs.tasks.drone_chase_env import DroneChaseEnv
from envs.sheeprl_wrapper import SheepRLCompatWrapper
from run_baselines import _TARGET_REL, _TARGET_VEL

def act(obs, gain, clip, lead):
    s=obs["state"]; rel=s[_TARGET_REL].astype(np.float64); vel=s[_TARGET_VEL].astype(np.float64)
    return np.clip(gain*(rel+lead*vel), -clip, clip).astype(np.float32)

cfg=EnvConfig.from_yaml('configs/target/chase_pursuit_big_v2.yaml')
raw=DroneChaseEnv(config=cfg); env=SheepRLCompatWrapper(raw)
def run(gain,clip,lead,n=40):
    c=0
    for ep in range(n):
        o,_=env.reset(seed=1000+ep); done=False
        while not done:
            o,r,term,trunc,info=env.step(act(o,gain,clip,lead)); done=term or trunc
        c+=bool(info.get("is_success",False))
    return c/n
print("baseline gain=0.4 clip=0.5 lead=1.0:", f"{run(0.4,0.5,1.0)*100:.0f}%")
for clip in [0.7,0.9,1.0]:
    print(f"gain=0.4 clip={clip} lead=1.0:", f"{run(0.4,clip,1.0)*100:.0f}%")
print("gain=0.6 clip=0.9 lead=1.0:", f"{run(0.6,0.9,1.0)*100:.0f}%")
print("gain=0.5 clip=0.9 lead=2.0:", f"{run(0.5,0.9,2.0)*100:.0f}%")
print("gain=0.6 clip=1.0 lead=1.5:", f"{run(0.6,1.0,1.5)*100:.0f}%")
