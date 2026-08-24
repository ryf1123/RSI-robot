"""Render a trained policy to mp4/gif. Retrains the design (cheap, deterministic
given the seed) and rolls out."""
import sys, json, numpy as np, imageio, mujoco, argparse
from rsi.env import Hopper
from rsi.inner import rollout, Normalizer, train_and_eval
from rsi.rewards import TERMS, TERM_NAMES
from rsi.design import Design
import rsi.inner as inner

def train_policy(design, seed, n_iters=120):
    """Same as train_and_eval but returns the policy."""
    d=Design(design); w=d.weight_vec; hp=d["hp"]
    rng=np.random.default_rng(seed)
    env=Hopper(term_height=hp["term_height"], max_steps=inner.MAX_STEPS, seed=seed)
    M=np.zeros((Hopper.act_dim,Hopper.obs_dim)); norm=Normalizer(Hopper.obs_dim)
    top=max(1,int(round(hp["top_frac"]*inner.N_DIRS)))
    active=None
    for it in range(n_iters):
        deltas=rng.standard_normal((inner.N_DIRS,*M.shape))
        rp=np.array([rollout(env,M+hp["noise_std"]*dl,norm,w)[0] for dl in deltas])
        rm=np.array([rollout(env,M-hp["noise_std"]*dl,norm,w)[0] for dl in deltas])
        o=np.argsort(-np.maximum(rp,rm))[:top]
        sr=np.concatenate([rp[o],rm[o]]).std()+1e-6
        M+=hp["step_size"]/(top*sr)*np.einsum("i,ijk->jk",rp[o]-rm[o],deltas[o])
    return M, norm, w, None

def record(design, seed, out, n_iters=120, fps=40):
    M,norm,w,active = train_policy(design,seed,n_iters)
    env=Hopper(term_height=0.7,max_steps=inner.MAX_STEPS,seed=seed+10_000)
    o=env.reset(); frames=[]
    # visual-only model: lights + checker floor + markers, identical physics
    from rsi.env import _XML_SRC
    vis=_XML_SRC.replace("<worldbody>", """<asset>
      <texture name="grid" type="2d" builtin="checker" rgb1=".45 .47 .52" rgb2=".58 .60 .65" width="300" height="300"/>
      <material name="gridm" texture="grid" texrepeat="24 6" reflectance="0"/>
      <texture name="sky" type="skybox" builtin="gradient" rgb1=".55 .62 .72" rgb2=".9 .92 .95" width="64" height="64"/>
    </asset><worldbody>
      <light pos="0 -2 4" dir="0 .4 -1" diffuse=".9 .9 .9" specular=".2 .2 .2"/>
      <light pos="3 -1 3" dir="-.5 .3 -1" diffuse=".35 .35 .4"/>""")
    vis=vis.replace('rgba="0.8 0.9 0.8 1"','material="gridm" rgba="1 1 1 1"')
    vis=vis.replace('rgba="0.6 0.7 0.9 1"','rgba="0.85 0.35 0.30 1"')
    vm=mujoco.MjModel.from_xml_string(vis); vd=mujoco.MjData(vm)
    r=mujoco.Renderer(vm, 360, 640)
    cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance=3.6; cam.elevation=-8; cam.azimuth=90
    while True:
        vd.qpos[:]=env.d.qpos; vd.qvel[:]=env.d.qvel; mujoco.mj_forward(vm,vd)
        cam.lookat[:]=[env.d.qpos[0],0,0.85]
        r.update_scene(vd, cam); frames.append(r.render())
        o,s,done,fell=env.step(M@norm(o))
        if done: break
    imageio.mimsave(out, frames, fps=fps)
    print(out, len(frames), "frames, x =", round(float(env.d.qpos[0]),2), "fell", fell)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--run"); ap.add_argument("--out"); ap.add_argument("--seed",type=int,default=None)
    a=ap.parse_args()
    h=[json.loads(l) for l in open(f"{a.run}/history.jsonl")]
    b=max(h,key=lambda x:x["fitness"])
    cfg=json.load(open(f"{a.run}/config.json"))
    print(a.run,"best fit",b["fitness"],Design(b["design"]).pretty())
    record(b["design"], a.seed if a.seed is not None else cfg["seed"], a.out)
