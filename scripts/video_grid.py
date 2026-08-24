"""Side-by-side rollouts with captions, so the reader can see what a design does."""
import json, argparse, numpy as np, imageio, mujoco
from scripts.render import train_policy
from rsi.env import Hopper, _XML_SRC
import rsi.inner as inner

VIS = _XML_SRC.replace("<worldbody>", """<asset>
 <texture name="grid" type="2d" builtin="checker" rgb1=".45 .47 .52" rgb2=".58 .60 .65" width="300" height="300"/>
 <material name="gridm" texture="grid" texrepeat="30 8"/>
 <texture name="sky" type="skybox" builtin="gradient" rgb1=".55 .62 .72" rgb2=".9 .92 .95" width="64" height="64"/>
</asset><worldbody>
 <light pos="0 -2 4" dir="0 .4 -1" diffuse=".95 .95 .95"/>
 <light pos="3 -1 3" dir="-.5 .3 -1" diffuse=".35 .35 .4"/>""").replace(
 'rgba="0.8 0.9 0.8 1"','material="gridm" rgba="1 1 1 1"').replace('rgba="0.6 0.7 0.9 1"','rgba="0.85 0.35 0.30 1"')

def panel(design, seed, W=420, H=340, n_iters=120):
    M,norm,w,_ = train_policy(design, seed, n_iters)
    env=Hopper(term_height=0.7, max_steps=inner.MAX_STEPS, seed=seed+10_000)
    vm=mujoco.MjModel.from_xml_string(VIS); vd=mujoco.MjData(vm)
    r=mujoco.Renderer(vm,H,W)
    cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance=3.8; cam.elevation=-8; cam.azimuth=90
    o=env.reset(); fr=[]
    while True:
        vd.qpos[:]=env.d.qpos; vd.qvel[:]=env.d.qvel; mujoco.mj_forward(vm,vd)
        cam.lookat[:]=[env.d.qpos[0],0,0.85]
        r.update_scene(vd,cam); fr.append(r.render())
        o,s,done,fell=env.step(M@norm(o))
        if done: break
    return fr, float(env.d.qpos[0]), fell

def grid(items, out, fps=40, W=420, H=340):
    panels=[]
    for lab, design, seed in items:
        f,x,fell = panel(design,seed,W,H); panels.append((lab,f,x,fell))
        print(f"  {lab}: {len(f)} frames, x={x:.2f}, fell={fell}", flush=True)
    T=max(len(f) for _,f,_,_ in panels)
    out_frames=[]
    for t in range(T):
        row=[]
        for lab,f,x,fell in panels:
            img=f[min(t,len(f)-1)].copy()
            if t>=len(f):  # frozen -> dim it
                img=(img*0.55).astype(np.uint8)
            img[:26]= (30,30,34)
            row.append(img)
        out_frames.append(np.concatenate(row,axis=1))
    imageio.mimsave(out, out_frames, fps=fps)
    print(out, len(out_frames), "frames | labels:", [p[0] for p in panels])

if __name__=="__main__":
    import sys
    which=sys.argv[1] if len(sys.argv)>1 else "decoy"
    if which=="decoy":
        from rsi.design import Design
        base=Design.zeros(); base["w"].update(fwd_vel=4.0, alive=0.25, height=2.0, upright=0.25)
        base["hp"].update(step_size=0.02, noise_std=0.02, top_frac=0.25, term_height=0.6)
        d1=Design(w=dict(base["w"]),hp=dict(base["hp"]))
        d2=Design(w=dict(base["w"]),hp=dict(base["hp"])); d2["w"]["stand_still"]=4.0
        d3=Design(w=dict(base["w"]),hp=dict(base["hp"])); d3["w"]["back_vel"]=4.0
        d4=Design(w=dict(base["w"]),hp=dict(base["hp"])); d4["w"]["foot_press"]=4.0
        grid([("clean",d1,0),("+stand_still",d2,0),("+back_vel",d3,0),("+foot_press",d4,0)],
             "docs/figs/decoys.gif", W=340, H=300)
