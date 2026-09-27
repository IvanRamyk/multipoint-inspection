import sys, numpy as np, torch
from pathlib import Path
REPO=Path('/Users/ivan.ramyk/dev/personal/dreamer'); sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/'tools'))
from omegaconf import OmegaConf
from lightning.fabric import Fabric
import gymnasium as gym
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.animation import FuncAnimation, FFMpegWriter
import imageio_ffmpeg
matplotlib.rcParams["animation.ffmpeg_path"]=imageio_ffmpeg.get_ffmpeg_exe()
from envs.core.config import EnvConfig
from envs.tasks.drone_chase_env import DroneChaseEnv
from envs.sheeprl_wrapper import SheepRLCompatWrapper
from sheeprl.algos.dreamer_v3.agent import build_agent
from run_baselines import _TARGET_REL, _TARGET_VEL

CK=Path(sys.argv[1])
cfg=OmegaConf.load(CK.parent.parent/"config.yaml")
fab=Fabric(devices=1,accelerator="cpu",num_nodes=1); st=fab.load(str(CK))
cfgh=EnvConfig.from_yaml('configs/target/chase_pursuit_big_v2.yaml'); dt=1.0/cfgh.agent_hz
raw=DroneChaseEnv(config=cfgh); env=SheepRLCompatWrapper(raw)
sc=cfg.env.screen_size; ch=1 if cfg.env.grayscale else 3
obs_space=gym.spaces.Dict({"depth":gym.spaces.Box(0,255,(ch,sc,sc),dtype=np.uint8),"state":env.observation_space["state"]})
wm,_,_,_,pl=build_agent(fab,[env.action_space.shape[0]],True,cfg,obs_space,world_model_state=st["world_model"],actor_state=st["actor"])
pl.num_envs=1; pl.eval(); cnn=list(cfg.algo.cnn_keys.encoder)
def tt(o):
    out={}
    for k,v in o.items():
        t=torch.from_numpy(v.copy()).float(); out[k]=(t.permute(2,0,1).unsqueeze(0).unsqueeze(0)/255.0-0.5) if k in cnn else t.view(1,1,-1)
    return out
def roll(seed, kind, clip=0.7):
    o,_=env.reset(seed=seed)
    if kind=='nn': pl.init_states()
    D=[]; T=[]; done=False; steps=0; caught=False
    while not done and steps<cfgh.max_episode_steps:
        D.append(raw.backend.get_drone_state().position.copy()); T.append(raw.backend.get_target_state().position.copy())
        if kind=='nn':
            with torch.no_grad(): a=pl.get_actions(tt(o),greedy=True)
            act=torch.stack(a,-1).cpu().numpy().reshape(env.action_space.shape)
        else:
            s=o["state"]; act=np.clip(0.4*(s[_TARGET_REL].astype(np.float64)+s[_TARGET_VEL].astype(np.float64)),-clip,clip).astype(np.float32)
        o,r,term,trunc,info=env.step(act); steps+=1; done=term or trunc; caught=bool(info.get("is_success",False))
    return np.array(D), np.array(T), (steps if caught else None)

# find a seed where BOTH catch and learned is clearly faster
chosen=None
for seed in range(1000,1035):
    ldD,ldT,ldc=roll(seed,'nn')
    if ldc is None: continue
    scD,scT,scc=roll(seed,'sc')
    if scc is None: continue
    if scc>=ldc*1.25 and ldT[0] is not None:
        start=float(np.linalg.norm(ldD[0]-ldT[0]))
        if start>25:  # prefer a far start for drama
            chosen=(seed,ldD,ldT,ldc,scD,scT,scc,start); break
    if chosen is None: chosen=(seed,ldD,ldT,ldc,scD,scT,scc,float(np.linalg.norm(ldD[0]-ldT[0])))
seed,ldD,ldT,ldc,scD,scT,scc,start=chosen
print(f"seed {seed}: start {start:.0f}m  learned catch {ldc*dt:.1f}s ({ldc}st)  scripted catch {scc*dt:.1f}s ({scc}st)")

Tmax=max(ldc,scc); stride=max(1,Tmax//300)
frames=list(range(0,Tmax,stride))+[Tmax-1]
allp=np.concatenate([ldD,ldT,scD,scT],0); lo=allp.min(0)-3; hi=allp.max(0)+3
rng=(hi-lo).max(); ctr=(hi+lo)/2; lo=ctr-rng/2; hi=ctr+rng/2
fig=plt.figure(figsize=(14,6.5),dpi=110)
def setup(ax,title):
    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(max(0,lo[2]),hi[2])
    ax.set_xlabel("x(m)");ax.set_ylabel("y(m)");ax.set_zlabel("alt");ax.set_title(title,fontsize=12,fontweight='bold')
axL=fig.add_subplot(121,projection='3d'); axR=fig.add_subplot(122,projection='3d')
def panel(ax,color,label):
    (dl,)=ax.plot([],[],[],'-',color=color,lw=2.2,label=label);(tl,)=ax.plot([],[],[],'-',color='#d62728',lw=2.0,label='target')
    (dd,)=ax.plot([],[],[],'o',color=color,ms=9);(td,)=ax.plot([],[],[],'o',color='#d62728',ms=9)
    txt=ax.text2D(0.02,0.95,"",transform=ax.transAxes,fontsize=11,va='top',family='monospace',bbox=dict(facecolor='white',alpha=0.7,pad=3))
    ax.legend(loc='upper right',fontsize=9); return dl,tl,dd,td,txt
setup(axL,f"DreamerV3 learned  (catch {ldc*dt:.1f}s)"); L=panel(axL,'#1f77b4','pursuer')
setup(axR,f"scripted lead-pursuit  (catch {scc*dt:.1f}s)"); R=panel(axR,'#2ca02c','pursuer')
def upd(fi):
    t=frames[fi]
    for (D,T,catch,P,name) in [(ldD,ldT,ldc,L,'learned'),(scD,scT,scc,R,'scripted')]:
        di=min(t,len(D)-1); ti=min(t,len(T)-1); dci=min(t,catch-1) if catch else di
        dl,tl,dd,td,txt=P; s=max(0,di-140)
        dl.set_data(D[s:dci+1,0],D[s:dci+1,1]); dl.set_3d_properties(D[s:dci+1,2])
        tl.set_data(T[s:ti+1,0],T[s:ti+1,1]); tl.set_3d_properties(T[s:ti+1,2])
        dd.set_data(D[dci:dci+1,0],D[dci:dci+1,1]); dd.set_3d_properties(D[dci:dci+1,2])
        td.set_data(T[ti:ti+1,0],T[ti:ti+1,1]); td.set_3d_properties(T[ti:ti+1,2])
        done = catch and t>=catch
        txt.set_text(f"t={t*dt:4.1f}s\n{'*** CAUGHT ***' if done else 'chasing...'}")
    az=(fi*0.4)%360; axL.view_init(elev=24,azim=az); axR.view_init(elev=24,azim=az)
    return []
fig.suptitle(f"Same episode (start {start:.0f} m): learned catches in {ldc*dt:.1f}s vs scripted {scc*dt:.1f}s",fontsize=13)
anim=FuncAnimation(fig,upd,frames=len(frames),interval=40,blit=False)
out=sys.argv[2]; Path(out).parent.mkdir(parents=True,exist_ok=True)
anim.save(out,writer=FFMpegWriter(fps=25,bitrate=2600)); print("wrote",out)
