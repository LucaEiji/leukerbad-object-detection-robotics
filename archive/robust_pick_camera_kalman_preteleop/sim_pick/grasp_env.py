"""SO-101 physical grasp environment. No hardware, no object attachment.

The learned action is a grasp pose. A numerical IK controller executes the
approach/close/lift sequence. Object pose observations are privileged simulator
state, not camera detections; this is not a real-robot deployment policy.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor/mujoco_menagerie/robotstudio_so101"


def scene_xml():
    root = ET.parse(VENDOR / "so101.xml").getroot()
    root.find("compiler").set("meshdir", str(VENDOR / "assets"))
    root.find("option").set("timestep", "0.005")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="table", type="plane", size="1 1 .02", rgba=".72 .76 .79 1", friction="1 .005 .0001")
    ET.SubElement(world, "light", pos=".2 -.4 1.2", dir="0 0 -1", diffuse=".8 .8 .8")
    ET.SubElement(world, "light", pos="-.3 .4 .8", dir="0 0 -1", diffuse=".5 .5 .5")
    gripper = world.find(".//body[@name='gripper']")
    ET.SubElement(gripper, "site", name="grasp_tcp", pos="0 0 -.092", size=".002", rgba="0 1 0 1", group="4")
    obj = ET.SubElement(world, "body", name="vape", pos=".23 0 .008")
    ET.SubElement(obj, "freejoint", name="vape_free")
    ET.SubElement(obj, "geom", name="vape_contact", type="capsule", fromto="-.038 0 0 .038 0 0", size=".008", mass=".025", rgba=".12 .22 .55 1", friction="1 .005 .0001", condim="6")
    ET.SubElement(obj, "geom", name="vape_mouthpiece", type="capsule", fromto=".031 0 0 .041 0 0", size=".0082", mass=".001", rgba=".04 .04 .05 1", contype="0", conaffinity="0")
    return ET.tostring(root, encoding="unicode")


class VapeGraspEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, randomize=True):
        self.model = mujoco.MjModel.from_xml_string(scene_xml())
        self.data = mujoco.MjData(self.model)
        self.ikdata = mujoco.MjData(self.model)
        self.sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_tcp")
        self.gbid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.obid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "vape")
        self.ogid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "vape_contact")
        self.low = self.model.jnt_range[:5, 0] + .01
        self.high = self.model.jnt_range[:5, 1] - .01
        self.action_space = gym.spaces.Box(-1, 1, (4,), np.float32)
        self.observation_space = gym.spaces.Box(-2, 2, (4,), np.float32)
        self.render_mode = render_mode
        self.randomize = randomize
        self.renderer = None
        self.frames = None
        self.frame_counter = 0
        self.trace = []
        self.home = np.array([0, -.3, .5, .7, 0, .5])

    def ik(self, xyz, yaw, initial=None):
        # Local gripper z points upwards, so its fingers point downwards.
        c, s = np.cos(yaw), np.sin(yaw)
        target_rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
        q = (self.data.qpos[:5] if initial is None else initial[:5]).copy()
        for _ in range(65):
            self.ikdata.qpos[:5] = q
            self.ikdata.qpos[5] = .5
            mujoco.mj_forward(self.model, self.ikdata)
            rot = self.ikdata.xmat[self.gbid].reshape(3, 3)
            orient = .5 * sum(np.cross(rot[:, i], target_rot[:, i]) for i in range(3))
            err = np.r_[xyz - self.ikdata.site_xpos[self.sid], .12 * orient]
            jp = np.zeros((3, self.model.nv)); jr = np.zeros_like(jp)
            mujoco.mj_jacSite(self.model, self.ikdata, jp, jr, self.sid)
            jac = np.vstack([jp[:, :5], .12 * jr[:, :5]])
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * .00005, err)
            q = np.clip(q + np.clip(delta, -.2, .2), self.low, self.high)
            if np.linalg.norm(err) < .0005:
                break
        return q

    def _contacts(self):
        names = []
        for contact in self.data.contact:
            if self.ogid in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == self.ogid else contact.geom1
                body = self.model.geom_bodyid[other]
                names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body))
        return sorted(set(names))

    def _record_trace(self, phase):
        self.trace.append({
            "time_s": round(float(self.data.time), 4),
            "phase": phase,
            "joint_rad": np.round(self.data.qpos[:5], 5).tolist(),
            "gripper": round(float(self.data.qpos[5]), 5),
            "tcp_m": np.round(self.data.site_xpos[self.sid], 5).tolist(),
            "object_pose_m": np.round(self.data.xpos[self.obid], 5).tolist(),
            "contacts": self._contacts(),
        })

    def _run(self, joints, gripper, seconds, phase="custom"):
        begin = self.data.ctrl.copy()
        target = np.r_[joints, gripper]
        steps = max(1, int(seconds / self.model.opt.timestep))
        self._record_trace(phase + ":start")
        for step in range(steps):
            alpha = (step + 1) / steps
            blend = alpha * alpha * (3 - 2 * alpha)
            self.data.ctrl[:] = begin + blend * (target - begin)
            mujoco.mj_step(self.model, self.data)
            if step % 20 == 0 or step == steps - 1:
                self._record_trace(phase)
            if self.frames is not None:
                self.frame_counter += 1
                if self.frame_counter % 7 == 0:
                    self.frames.append(self.render())
        self._record_trace(phase + ":end")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.trace = []
        self.objxy = np.array([.23, 0.])
        if self.randomize:
            self.objxy += self.np_random.uniform(-.005, .005, 2)
        self.yaw = float(self.np_random.uniform(-1.0, 1.0)) if self.randomize else 0.
        self.data.qpos[6:9] = [*self.objxy, .0085]
        self.data.qpos[9:13] = [np.cos(self.yaw / 2), 0, 0, np.sin(self.yaw / 2)]
        self.data.qpos[:6] = self.home
        self.data.ctrl[:] = self.home
        mujoco.mj_forward(self.model, self.data)
        home = self.ik(np.array([.20, 0, .17]), np.pi / 2, initial=self.home)
        self.data.qpos[:5] = home
        self.data.ctrl[:5] = home
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        self.start_height = float(self.data.xpos[self.obid, 2])
        obs = np.array([*(self.objxy - [.23, 0]) * 100, np.sin(2*self.yaw), np.cos(2*self.yaw)], np.float32)
        return obs, {}

    def step(self, action):
        action = np.clip(np.asarray(action), -1, 1)
        xyz = np.array([.23 + .035 * action[0], .035 * action[1], .018 + .009 * action[2]])
        yaw = np.pi / 2 + 1.05 * action[3]
        above = xyz + [0, 0, .10]
        qa = self.ik(above, yaw)
        qg = self.ik(xyz, yaw, initial=qa)
        ql = self.ik(xyz + [0, 0, .10], yaw, initial=qg)
        self._run(qa, .55, .65, "approach")
        self._run(qg, .55, .65, "descend")
        self._run(qg, -.14, .55, "close")
        self._run(ql, -.14, 1.0, "lift")
        self._run(ql, -.14, .4, "hold")
        height = float(self.data.xpos[self.obid, 2])
        lift = height - self.start_height
        contacts = self._contacts()
        held = "gripper" in contacts and "moving_jaw_so101_v1" in contacts
        success = lift > .04 and held
        position_error = np.linalg.norm(xyz[:2] - self.objxy)
        alignment = np.sin(yaw - self.yaw) ** 2
        shaping = 2 * np.exp(-position_error / .018) * (.3 + .7 * alignment)
        reward = shaping + 25 * np.clip(lift, 0, .12) + 8 * success
        info = dict(success=bool(success), lift=lift, held=bool(held), reward=float(reward), grasp_xyz=xyz.tolist(), grasp_yaw=float(yaw))
        obs = np.array([*(self.objxy - [.23, 0]) * 100, np.sin(2*self.yaw), np.cos(2*self.yaw)], np.float32)
        return obs, float(reward), True, False, info

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.18, 0, .15]
        camera.distance = .85
        camera.azimuth = 135
        camera.elevation = -25
        option = mujoco.MjvOption()
        option.geomgroup[3:] = 0
        self.renderer.update_scene(self.data, camera=camera, scene_option=option)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
