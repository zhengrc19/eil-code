from gym_robotics.envs.fetch_env import FetchEnv
import os.path as osp
import numpy as np

MODEL_XML_PATH = osp.join(
    osp.dirname(osp.abspath(__file__)),
    "assets",
    "fetch",
    "push.xml"
)


class PushEnv(FetchEnv):
    def __init__(self, reward_type="sparse"):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        super().__init__(
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=40,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        self._max_episode_steps = 60
    
    def _reset_sim(self):
        # self.initial_gripper_xpos[0] -= 0.5
        # import ipdb
        # ipdb.set_trace()
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_xpos = self.np_random.uniform(-0.1, 0.1, size=2) + np.array([1.34, 0.75])
            object_qpos[:2] = object_xpos.tolist()
            object_qpos[1] = 0.90
            object_qpos[2] = 0.42
            # object_qpos = [1.34, 0.75, 0.45, 1.0, 0.0, 0.0, 0.0]
            # object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1  # 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.
    
    def _sample_goal(self):
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        goal = np.array([1.45, 1, 0.4]) + self.np_random.uniform(-0.05, 0.05, size=3)
        goal[2] = 0.42  # 0.7
        goal[1] = 1.05
        goal[0] = object_qpos[0]
        return goal.copy()