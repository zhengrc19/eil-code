""" Dataset generation for series of fetch envs """

import os
import os.path as osp
from typing import Dict, List, Union

import gym
import numpy as np
from numpy.linalg import norm
from PIL import Image
from tqdm import tqdm
from demo_generation.Push import PushEnv
from demo_generation.ZoomDevReach import ZoomDevReachEnv
from demo_generation.Stir import StirEnv


class DemoGeneration:
    """ Data generation for Fetch Env """

    def __init__(self,
                 name: str,
                 phase: str = None,
                 perfect: str = None,
                 render: bool = False,
                 output_dir: str = '/tmp/eil_demos',
                 seed: int = 1) -> None:
        if phase and phase not in ['train', 'valid']:
            raise ValueError(f"Phase {phase} not available; "
                             "can only be 'train' or 'valid'. ")
        if perfect and perfect not in ['perfect', 'imperfect']:
            raise ValueError(f"Perfectness {perfect} not available; "
                             "can only be 'perfect' or 'imperfect'. ")
        self.phases = [phase] if phase else ['train', 'valid']
        self.perfects = [perfect] if perfect else ['imperfect', 'perfect']
        self.phase = None
        self.perfect = None
        self.render = render
        self.env: gym.Env = None
        self.timeStep = 0
        self.actions = []
        self.obs = []
        self.lens = []
        self.devs = []
        self.states = []
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.name = name
        self.seed = seed
        self.dev_distance = -1
        self.dev_dir: np.ndarray = None
        self.goal: np.ndarray = None
        self.dim_grip_pos = 3
        self.dim_object_pos = 3
        self.dim_object_rel_pos = 3
        self.dim_gripper_state = 3
        self.dim_object_rot = 3
        self.dim_object_velp = 3
        self.dim_object_velr = 3
        self.dim_grip_velp = 2
        self.dim_gripper_vel = 2
        self.dev_steps = 8

    def init_phase(self, acs, obs, imgs, devs, states) -> None:
        """ Initial phase in task policy """
        pass

    def before_dev(self, acs, obs, imgs, devs, states) -> None:
        """ Phase before deviation phase in task policy """
        pass

    def deviate(self, acs, obs, imgs, devs, states) -> None:
        """ Deviation phase """
        pass

    def after_dev(self, acs, obs, imgs, devs, states) -> None:
        """ Final phase after deviation """
        pass

    def sample_dev_distance(self, last_obs: dict) -> float:
        """ Sample a random distance to start deviation """
        pass

    def sample_dev_dir(self, last_obs: dict) -> np.ndarray:
        """ Sample a random deviation direction vector """
        pass

    def success(self, last_obs, img_count) -> bool:
        grip_pos = self.get_grip_pos(last_obs)
        return norm(self.goal - grip_pos) <= 0.05 and img_count > 40

    def get_grip_pos(self, obs: dict) -> np.ndarray:
        base = 0
        return obs['observation'][base:base + self.dim_grip_pos]

    def get_object_pos(self, obs: dict) -> np.ndarray:
        base = self.dim_grip_pos
        return obs['observation'][base:base + self.dim_object_pos]

    def get_object_rel_pos(self, obs: dict) -> np.ndarray:
        base = self.dim_grip_pos + self.dim_object_pos
        return obs['observation'][base:base + self.dim_object_rel_pos]

    def get_gripper_state(self, obs: dict) -> np.ndarray:
        base = np.sum([
            self.dim_grip_pos,
            self.dim_object_pos,
            self.dim_object_rel_pos,
        ])
        return obs['observation'][base:base + self.dim_gripper_state]

    def get_grip_velp(self, obs: dict) -> np.ndarray:
        base = np.sum([
            self.dim_grip_pos,
            self.dim_object_pos,
            self.dim_object_rel_pos,
            self.dim_gripper_state,
            self.dim_object_rot,
            self.dim_object_velp,
            self.dim_object_velr,
        ])
        return obs['observation'][base:base + self.dim_grip_velp]

    def get_gripper_vel(self, obs: dict) -> np.ndarray:
        base = np.sum([
            self.dim_grip_pos,
            self.dim_object_pos,
            self.dim_object_rel_pos,
            self.dim_gripper_state,
            self.dim_object_rot,
            self.dim_object_velp,
            self.dim_object_velr,
            self.dim_grip_velp,
        ])
        return obs['observation'][base:base + self.dim_gripper_vel]

    def get_goal(self, obs: dict) -> np.ndarray:
        """ Get coordinates of goal (red spot) """
        return obs['desired_goal']

    def get_state(self, obs: dict) -> np.ndarray:
        return np.concatenate([
            self.get_grip_pos(obs),
            self.get_gripper_state(obs),
            self.get_grip_velp(obs),
            self.get_gripper_vel(obs)
        ])

    def get_action(self, intended: Union[list, np.ndarray]) -> np.ndarray:
        action = np.zeros((4, ))
        for i in range(len(intended)):
            action[i] = intended[i]
        return action

    def get_noise(self, action: np.ndarray, last_obs: dict) -> np.ndarray:
        pass

    def step_env(self,
                 action: np.ndarray,
                 acs: list,
                 obs: list,
                 devs: list,
                 states: list,
                 gaussian_noise: bool = False,
                 is_dev: bool = False) -> Dict[str, np.ndarray]:
        noise = np.zeros((4, ))
        if gaussian_noise \
            and self.phase == 'train' \
            and len(self.lens) \
            and np.random.random() > 0.5:
            noise[:3] += self.get_noise(action, obs[-1])[:3]

        new_obs, _, _, _ = self.env.step(action + noise)
        acs.append(action)
        obs.append(new_obs)
        states.append(self.get_state(new_obs))
        devs.append(1 if is_dev else 0)
        self.timeStep += 1

        return new_obs

    def go_to_goal(self, last_obs, folder) -> int:
        epi_acs = []
        epi_obs = []
        epi_imgs: List[Image.Image] = []
        epi_devs = []
        epi_states = []
        self.goal = self.get_goal(last_obs)
        self.timeStep = 0  # count the total number of timesteps
        epi_obs.append(last_obs)
        epi_states.append(self.get_state(last_obs))

        # step through
        self.init_phase(epi_acs, epi_obs, epi_imgs, epi_devs, epi_states)
        if self.perfect == 'imperfect' and len(self.lens):
            self.before_dev(epi_acs, epi_obs, epi_imgs, epi_devs, epi_states)
            self.deviate(epi_acs, epi_obs, epi_imgs, epi_devs, epi_states)
        self.after_dev(epi_acs, epi_obs, epi_imgs, epi_devs, epi_states)

        img_count = len(epi_imgs)
        last_obs = epi_obs[-1]
        # only allow video to be saved if success and more than 40 frames
        if self.success(last_obs, img_count):
            self.actions.extend(epi_acs)
            self.devs.extend(epi_devs)
            self.obs.extend(epi_obs)
            epi_states.pop()
            self.states.extend(epi_states)
            for i, img in enumerate(epi_imgs):
                img.save(osp.join(folder, f'{i:04d}.jpg'))
            return img_count
        return 0

    def gen_image(self, img_list: List[Image.Image]) -> None:
        if self.render:
            self.env.render()
        else:
            img = self.env.render(mode='rgb_array')
            assert (img.shape == (500, 500, 3))
            img = Image.fromarray(img)
            img_list.append(img)

    def generate(self) -> None:
        for self.phase in self.phases:
            numItr = 40 if self.phase == 'train' else 16
            for self.perfect in self.perfects:
                for l in [
                        self.actions, self.obs, self.lens, self.devs,
                        self.states
                ]:
                    l.clear()
                self.env.seed(self.seed)
                # print(f"Reset! {self.perfect}_demo_{self.phase}")
                root_dir = osp.join(self.output_dir, self.name,
                                    f'{self.perfect}_demo_{self.phase}')
                pbar = tqdm(total=numItr,
                            desc=f"{self.name} {self.perfect}_demo_{self.phase}")
                while len(self.lens) < numItr:
                    # print(f"\rITERATION {len(self.lens)}", end='')
                    folder = osp.join(root_dir, 'images',
                                      f'{len(self.lens):04d}')
                    os.makedirs(folder, exist_ok=True)
                    obs = self.env.reset()
                    seq_len = self.go_to_goal(obs, folder)
                    if seq_len:
                        self.lens.append(seq_len)
                        pbar.update(1)
                pbar.close()

                print(np.sum(self.lens), "total", np.sum(self.devs), "dev")
                # print(self.lens)
                # print(self.devs)
                np.save(osp.join(root_dir, 'labels.npy'), {
                    'actions': np.array(self.actions),
                    'seq_lens': np.array(self.lens),
                    'is_deviation': np.array(self.devs),
                    'states': np.array(self.states)
                }, allow_pickle=True)
                new_dir = "/tmp/xirl_format_datasets"
                new_dir = osp.join(new_dir, self.phase)
                if self.perfect == 'perfect':
                    new_dir = osp.join(new_dir, self.name + '_perfect')
                else:
                    new_dir = osp.join(new_dir, self.name)
                if not osp.exists(new_dir):
                    # os.mkdir(new_dir)
                    os.makedirs(new_dir)
                    os.system(
                        f'cp -r {osp.join(root_dir, "images", "*")} {new_dir}')
                    os.system(
                        f'cp {osp.join(root_dir, "labels.npy")} {new_dir}')


class ReachDemoGeneration(DemoGeneration):
    """ Data generation for reach task in Fetch Env """

    def __init__(self,
                 name: str,
                 phase: str = None,
                 perfect: str = None,
                 render: bool = False,
                 output_dir: str = '/tmp/eil_demos',
                 seed: int = 1) -> None:
        super().__init__(name, phase, perfect, render, output_dir, seed)
        self.env = ZoomDevReachEnv()
        self.dim_object_pos = 0
        self.dim_object_rel_pos = 0
        self.dim_object_rot = 0
        self.dim_object_velp = 0
        self.dim_object_velr = 0

    def get_noise(self, action: np.ndarray, last_obs: dict) -> np.ndarray:
        return np.random.normal(0.0, np.max(np.abs(action[:3])) / 2, size=4)

    def before_dev(self, acs, obs, imgs, devs, states) -> None:
        last_obs = obs[-1]
        self.dev_distance = self.sample_dev_distance(last_obs)
        self.dev_dir = self.sample_dev_dir(last_obs)
        grip_pos = self.get_grip_pos(last_obs)
        while (norm(self.goal - grip_pos) >= self.dev_distance
               and self.timeStep <= self.env._max_episode_steps):
            self.gen_image(imgs)
            action = self.get_action((self.goal - grip_pos) *
                                     (2 + int(self.perfect == 'imperfect')))
            new_obs = self.step_env(action,
                                    acs,
                                    obs,
                                    devs,
                                    states,
                                    gaussian_noise=True)
            grip_pos = self.get_grip_pos(new_obs)

    def deviate(self, acs, obs, imgs, devs, states) -> None:
        for _ in range(self.dev_steps):
            self.gen_image(imgs)
            self.step_env(self.dev_dir, acs, obs, devs, states, is_dev=True)
        back = self.dev_dir
        back[:3] *= -1  # do not change gripper claw vel
        for _ in range(self.dev_steps):
            self.gen_image(imgs)
            self.step_env(back, acs, obs, devs, states, is_dev=True)

    def after_dev(self, acs, obs, imgs, devs, states) -> None:
        last_obs = obs[-1]
        grip_pos = self.get_grip_pos(last_obs)
        while norm(self.goal - grip_pos) >= 0.001:
            self.gen_image(imgs)
            action = self.get_action((self.goal - grip_pos) *
                                     (2 + int(self.perfect == 'imperfect')))
            new_obs = self.step_env(action,
                                    acs,
                                    obs,
                                    devs,
                                    states,
                                    gaussian_noise=True)
            grip_pos = self.get_grip_pos(new_obs)
            if self.timeStep > self.env._max_episode_steps:
                break

    def sample_dev_dir(self, last_obs: dict) -> np.ndarray:
        dir = np.random.normal(0, 1, size=(4, ))
        dir[1] = dir[1] * 10 if dir[1] < 0 else dir[1] * -10
        dir[2] /= 10
        dir[3] = 0
        return dir

    def sample_dev_distance(self, last_obs: dict) -> float:
        distance = norm(self.goal - self.get_grip_pos(last_obs))
        distraction = np.random.uniform(0, distance)
        return distraction


class StirDemoGeneration(DemoGeneration):
    """ Data generation for stir task in Fetch Env """

    def __init__(self,
                 name: str,
                 phase: str = None,
                 perfect: str = None,
                 render: bool = False,
                 output_dir: str = '/tmp/eil_demos',
                 seed: int = 1) -> None:
        super().__init__(name, phase, perfect, render, output_dir, seed)
        self.env = StirEnv()
        self.center = np.array([1.34, 0.75, 0.3])
        self.theta = 0
        self.angular_vel = 0.07
        self.radius = 0.1
        self.dim_object_pos = 0
        self.dim_object_rel_pos = 0
        self.dim_object_rot = 0
        self.dim_object_velp = 0
        self.dim_object_velr = 0

    def init_phase(self, acs, obs, imgs, devs, states) -> None:
        last_obs = obs[-1]
        grip_pos = self.get_grip_pos(last_obs)
        start = self.center + np.array([self.radius, 0, 0])
        while (norm(start - grip_pos) > 0.3
               and self.timeStep <= self.env._max_spisode_steps):
            self.gen_image(imgs)
            action = self.get_action((start - grip_pos) * 10)
            new_obs = self.step_env(action, acs, obs, devs, states)
            grip_pos = self.get_grip_pos(new_obs)

    def before_dev(self, acs, obs, imgs, devs, states) -> None:
        new_obs = obs[-1]
        self.dev_distance = self.sample_dev_distance(new_obs)
        grip_pos = self.get_grip_pos(new_obs)
        while self.timeStep <= self.dev_distance:
            self.gen_image(imgs)
            self.theta += self.angular_vel
            new_goal = self.center + np.array([
                self.radius * np.cos(self.theta),
                self.radius * np.sin(self.theta), 0.2
            ])
            action = self.get_action((new_goal - grip_pos) * 5)
            new_obs = self.step_env(action, acs, obs, devs, states)
            grip_pos = self.get_grip_pos(new_obs)
        self.dev_dir = self.sample_dev_dir(new_obs)

    def deviate(self, acs, obs, imgs, devs, states) -> None:
        for _ in range(self.dev_steps):
            self.gen_image(imgs)
            self.step_env(self.dev_dir, acs, obs, devs, states, is_dev=True)
        back = self.dev_dir
        back[:3] *= -1  # do not change gripper claw vel
        for _ in range(self.dev_steps):
            self.gen_image(imgs)
            self.step_env(back, acs, obs, devs, states, is_dev=True)

    def after_dev(self, acs, obs, imgs, devs, states) -> None:
        last_obs = obs[-1]
        grip_pos = self.get_grip_pos(last_obs)
        while True:
            self.gen_image(imgs)
            self.theta += self.angular_vel
            new_goal = self.center + np.array([
                self.radius * np.cos(self.theta),
                self.radius * np.sin(self.theta), 0.2
            ])
            action = self.get_action((new_goal - grip_pos) * 5)
            new_obs = self.step_env(action, acs, obs, devs, states)
            grip_pos = self.get_grip_pos(new_obs)
            if self.timeStep > self.env._max_episode_steps:
                break

    def sample_dev_dir(self, last_obs: dict) -> np.ndarray:
        grip_pos = self.get_grip_pos(last_obs)
        return np.insert((self.goal - grip_pos) * 3, 3, 0)

    def sample_dev_distance(self, last_obs: dict) -> float:
        return np.random.uniform(
            self.timeStep, self.env._max_episode_steps - 2 * self.dev_steps)

    def success(self, last_obs, img_count) -> bool:
        return True


class PushDemoGeneration(DemoGeneration):
    """ Data generation for push task in Fetch Env """

    def __init__(self,
                 name: str,
                 phase: str = None,
                 perfect: str = None,
                 render: bool = False,
                 output_dir: str = '/tmp/eil_demos',
                 seed: int = 1) -> None:
        super().__init__(name, phase, perfect, render, output_dir, seed)
        self.env = PushEnv()

    def get_noise(self, action: np.ndarray, last_obs: dict) -> np.ndarray:
        object_pos = self.get_object_pos(last_obs)
        return np.random.normal(0, np.abs(self.goal - object_pos) / 10)

    def init_phase(self, acs, obs, imgs, devs, states) -> None:
        new_obs = obs[-1]
        self.dev_distance = self.sample_dev_distance(new_obs)
        object_pos = self.get_object_pos(new_obs)
        grip_pos = self.get_grip_pos(new_obs)
        while (abs(grip_pos[0] - object_pos[0]) >= 0.01
               and self.timeStep <= self.dev_distance
               and self.timeStep <= self.env._max_episode_steps):
            self.gen_image(imgs)
            action = self.get_action([(object_pos[0] - grip_pos[0]) * 3])
            new_obs = self.step_env(action, acs, obs, devs, states)
            object_pos = self.get_object_pos(new_obs)
            grip_pos = self.get_grip_pos(new_obs)

    def before_dev(self, acs, obs, imgs, devs, states) -> None:
        new_obs = obs[-1]
        # self.dev_distance = self.sample_dev_distance(new_obs)
        object_pos = self.get_object_pos(new_obs)
        while (self.timeStep <= self.dev_distance
            #    and norm(self.goal - object_pos) >= self.dev_distance
               and self.timeStep <= self.env._max_episode_steps):
            self.gen_image(imgs)
            orig_action = self.goal - object_pos
            if self.perfect == 'imperfect':
                action = self.get_action(orig_action * 1.5)
            else:
                action = self.get_action(orig_action)
            if not len(self.lens):
                action = self.get_action(orig_action * 1.1)
            new_obs = self.step_env(action,
                                    acs,
                                    obs,
                                    devs,
                                    states,
                                    gaussian_noise=True)
            object_pos = self.get_object_pos(new_obs)

    def deviate(self, acs, obs, imgs, devs, states) -> None:
        last_obs = obs[-1]
        self.env.step([0, 0, 0, 0])
        self.dev_dir = self.sample_dev_dir(last_obs)
        for _ in range(self.dev_steps):
            self.gen_image(imgs)
            self.step_env(self.dev_dir, acs, obs, devs, states, is_dev=True)
        back = self.dev_dir
        back[:3] *= -1
        for _ in range(self.dev_steps):
            self.gen_image(imgs)
            self.step_env(back, acs, obs, devs, states, is_dev=True)

    def after_dev(self, acs, obs, imgs, devs, states) -> None:
        new_obs = obs[-1]
        object_pos = self.get_object_pos(new_obs)
        grip_pos = self.get_grip_pos(new_obs)
        while norm(self.goal - object_pos) >= 0.01:
            self.gen_image(imgs)
            if abs(grip_pos[0] - object_pos[0]) < 0.01:
                orig_action = self.goal - object_pos
                if self.perfect == 'imperfect':
                    action = self.get_action(orig_action * 1.5)
                else:
                    action = self.get_action(orig_action)
                if not len(self.lens):
                    action = self.get_action(orig_action * 1.1)
                new_obs = self.step_env(action,
                                        acs,
                                        obs,
                                        devs,
                                        states,
                                        gaussian_noise=True)
                object_pos = self.get_object_pos(new_obs)
            else:
                action = self.get_action([(object_pos[0] - grip_pos[0]) * 3])
                new_obs = self.step_env(action, acs, obs, devs, states)
                object_pos = self.get_object_pos(new_obs)
                grip_pos = self.get_grip_pos(new_obs)
            if self.timeStep > self.env._max_episode_steps:
                break

    def sample_dev_dir(self, last_obs: dict) -> np.ndarray:
        grip_pos = self.get_grip_pos(last_obs)
        dev_dir = np.random.normal(0, 0.2, size=(4, ))
        dev_dir[3] = 0
        dev_dir[2] = 0
        dev_dir[1] = -abs(dev_dir[1])
        dev_dir[0] = ((1.5 if dev_dir[0] > 0 else 1.15) - grip_pos[0]) / 2
        dev_dir[:3] *= 3
        return dev_dir

    def sample_dev_distance(self, last_obs: dict) -> float:
        return np.random.randint(self.timeStep + 1, self.env._max_episode_steps)
    
    def success(self, last_obs, img_count) -> bool:
        object_pos = self.get_object_pos(last_obs)
        return norm(self.goal - object_pos) < 0.05 and img_count > 40


if __name__ == '__main__':
    reach = ReachDemoGeneration("reach_0529", seed=10)
    # stir = StirDemoGeneration("stir_0219", seed=10)
    # push = PushDemoGeneration("push_0225", seed=10)
    reach.generate()
    # stir.generate()
    # push.generate()