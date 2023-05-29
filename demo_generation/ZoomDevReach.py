from gym_robotics.envs.fetch_env import FetchEnv
import os.path as osp
import numpy as np

MODEL_XML_PATH = osp.join(
    osp.dirname(osp.abspath(__file__)),
    "assets",
    "fetch",
    "reach.xml"
)


class ZoomDevReachEnv(FetchEnv):
    def __init__(self, reward_type="sparse"):
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        super().__init__(
            model_path=MODEL_XML_PATH,
            n_substeps=40,
            gripper_extra_height=0.2,
            block_gripper=True,
            has_object=False,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        self._max_episode_steps = 50
    
    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id("robot0:gripper_link")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1  # 2.5
        self.viewer.cam.azimuth = 132.0
        self.viewer.cam.elevation = -14.0
    
    def _sample_goal(self):
        return np.array([1.45, 1, 0.5]) + self.np_random.uniform(-0.05, 0.05, size=3)