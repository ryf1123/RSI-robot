"""Planar hopper task. The *task metric* (forward distance) is fixed and is never
what the inner optimizer sees -- the inner optimizer only ever sees the *designed*
reward. Keeping these two separate is the whole point of the project."""
import numpy as np, mujoco, os

XML = os.path.join(os.path.dirname(__file__), "tasks", "hopper.xml")
_XML_SRC = open(XML).read()

# joint ranges (deg) from the xml, used by the joint-limit term
JOINT_RANGE = np.deg2rad(np.array([[-150.0, 0.0], [-150.0, 0.0], [-45.0, 45.0]]))
INIT_HEIGHT = 1.25


class Hopper:
    """frame_skip 5 -> 100 Hz control. obs = [z, pitch, 3 joints, 6 vel] = 11."""
    frame_skip = 3
    obs_dim = 11
    act_dim = 3

    def __init__(self, term_height=0.7, term_angle=0.2, max_steps=200, seed=0):
        self.m = mujoco.MjModel.from_xml_string(_XML_SRC)
        self.d = mujoco.MjData(self.m)
        self.term_height = term_height
        self.term_angle = term_angle
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.dt = self.m.opt.timestep * self.frame_skip

    # ---- state helpers -------------------------------------------------
    @property
    def height(self):
        return INIT_HEIGHT + self.d.qpos[1]

    @property
    def pitch(self):
        return self.d.qpos[2]

    def _obs(self):
        return np.concatenate([[self.height - INIT_HEIGHT, self.pitch],
                               self.d.qpos[3:], np.clip(self.d.qvel, -10, 10)])

    def reset(self, noise=5e-3):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] += self.rng.uniform(-noise, noise, self.m.nq)
        self.d.qvel[:] += self.rng.uniform(-noise, noise, self.m.nv)
        mujoco.mj_forward(self.m, self.d)
        self.t = 0
        self.prev_u = np.zeros(self.act_dim)
        self.prev_x = self.d.qpos[0]
        self.contact_steps = 0
        return self._obs()

    def step(self, u):
        u = np.clip(u, -1, 1)
        self.prev_x = self.d.qpos[0]
        self.d.ctrl[:] = u
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.m, self.d)
        self.t += 1
        # foot contact
        foot_z = self.d.geom_xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "foot_geom")][2]
        self.foot_z = foot_z
        if foot_z < 0.08:
            self.contact_steps += 1
        s = dict(x=self.d.qpos[0], dx=(self.d.qpos[0] - self.prev_x) / self.dt,
                 z=self.height, pitch=self.pitch, qpos=self.d.qpos.copy(),
                 qvel=self.d.qvel.copy(), u=u, prev_u=self.prev_u, foot_z=foot_z,
                 airborne=foot_z > 0.12)
        self.prev_u = u
        fell = (self.height < self.term_height) or (abs(self.pitch) > self.term_angle) \
            or not np.isfinite(self.d.qpos).all()
        done = fell or self.t >= self.max_steps
        return self._obs(), s, done, fell
