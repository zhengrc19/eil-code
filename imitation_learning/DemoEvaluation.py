from copy import deepcopy
import os
import os.path as osp
from time import time
from typing import Dict, List, Tuple, Type

from PIL import Image
from tqdm import trange
import numpy as np
from demo_generation.DemoGeneration import DemoGeneration, PushDemoGeneration, ReachDemoGeneration, StirDemoGeneration
from imitation_learning.ImitateModel import ImitateModel, create_model
import torch
from torchvision import transforms


class DemoEvaluation:
    """ Evaluation in fetch env. """

    def __init__(self,
                 Task: Type[DemoGeneration],
                 train_id: str,
                 ckpts: List[int],
                 win_len: int = 1,
                 render_dir: str = '/tmp/resnet_train_results',
                 num_itrs: int = 50,
                 render: bool = False,
                 seed: int = 2) -> None:
        self.Task = Task(name=train_id, render=render, seed=seed)
        self.train_id = train_id
        self.ckpts = ckpts
        self.win_len = win_len
        self.render_dir = render_dir
        self.video_dir: str = ''
        self.num_itrs = num_itrs
        self.init_elem_value = 100000000
        self.min_dists: List[float] = []
        self.img_queue: List[Image.Image] = []
        self.raw_data: Dict[int, List[float]] = {}
        self.timeStep: int = 0
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model: ImitateModel = None
        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5452, 0.7267, 0.6137],
                                 [0.2064, 0.2000, 0.1730])
        ])
        self.begin = time()
        ckpt_name = osp.join(render_dir, train_id,
                             f'checkpoint_{train_id}_{ckpts[0]}.pt')
        ckpt = torch.load(ckpt_name)
        self.use_state = ckpt['use_state']
        self.state_dim = ckpt['state_dim']
        self.model, _, _, _ = create_model(self.win_len,
                                           self.device,
                                           use_state=self.use_state,
                                           state_dim=self.state_dim)

    def evaluate(self) -> Dict[int, List[float]]:
        self.model.eval()
        self.Task.env.seed(self.Task.seed)
        self.raw_data.clear()
        self.raw_data["raw_data"] = {}
        ckpt_obs = []
        ckpt_acs = []
        for ckpt_idx in self.ckpts:
            ckpt_name = osp.join(self.render_dir, self.train_id,
                                 f'checkpoint_{self.train_id}_{ckpt_idx}.pt')
            ckpt = torch.load(ckpt_name)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model.eval()
            self.min_dists.clear()
            ckpt_obs.clear()
            ckpt_acs.clear()
            for self.vidname in trange(self.num_itrs,
                                       desc=f"{self.train_id} CKPT {ckpt_idx}",
                                       leave=False):
                if self.Task.render:
                    self.video_dir = osp.join(self.render_dir, self.train_id,
                                              "eval_results", str(ckpt_idx),
                                              f"{self.vidname:04d}")
                    os.makedirs(self.video_dir, exist_ok=True)
                obs = self.Task.env.reset()
                with torch.set_grad_enabled(False):
                    epi_obs, epi_acs = self.imitate_go_to_goal(obs)
                ckpt_obs.append(epi_obs)
                ckpt_acs.append(epi_acs)
                
            # duration = time() - self.begin
            # print("at {}h {}m {}s, finished :".format(
            #     int(duration // 3600), int((duration % 3600) // 60),
            #     int(duration % 60)),
            #       end='')
            # print(ckpt_name)
            # print(min(self.min_dists), self.min_dists)
            # filename = osp.join(self.render_dir, self.train_id,
            #                     f"results_{ckpt_idx}.txt")
            # with open(filename, 'w') as f:
            #     print(ckpt_name, file=f, end='\t')
            #     print(*(self.min_dists), sep='\t', file=f)
            self.raw_data[ckpt_idx] = np.array(self.min_dists)
            self.raw_data["raw_data"][ckpt_idx] = {
                "observations": ckpt_obs,
                "actions": ckpt_acs 
            }
        return self.raw_data

    def infer_action(self, obs) -> np.ndarray:
        img = self.Task.env.render(mode='rgb_array')
        if self.Task.render:
            real_img = Image.fromarray(img)
            real_img.save(osp.join(self.video_dir, f'{self.timeStep:04d}.jpg'))

        img = self.trans(img)
        if len(self.img_queue):
            self.img_queue.pop(0)
            self.img_queue.append(img)
        else:
            self.img_queue = [img for _ in range(self.win_len)]
        imgs = torch.stack(self.img_queue)
        imgs = torch.unsqueeze(imgs, 0)
        imgs = imgs.to(self.device)
        if self.use_state:
            state = torch.tensor(self.Task.get_state(obs)).float()
            state = torch.unsqueeze(state.to(self.device), 0)
            inputs = {"images": imgs, "state": state}
        else:
            inputs = imgs
        action = self.model(inputs)
        action = torch.reshape(action, (4, )).cpu().numpy()
        return action

    def update_result(self, obs) -> None:
        grip_pos = self.Task.get_grip_pos(obs)
        goal = self.Task.get_goal(obs)
        if np.linalg.norm(goal - grip_pos) < self.min_dists[-1]:
            self.min_dists[-1] = np.linalg.norm(goal - grip_pos)

    def imitate_go_to_goal(self, obs) -> Tuple[np.ndarray, np.ndarray]:
        self.min_dists.append(self.init_elem_value)
        self.timeStep = 0
        self.img_queue.clear()
        episode_obs = []
        episode_acs = []
        if isinstance(self, StirDemoEvaluation):
            self.radii.clear()

        while True:
            action = self.infer_action(obs)

            episode_obs.append(obs)
            episode_acs.append(action)

            obs, _, _, _ = self.Task.env.step(action)
            self.timeStep += 1

            self.update_result(obs)

            if self.timeStep >= self.Task.env._max_episode_steps * 1.2:
                break
        
        return np.array(episode_obs, dtype=object), np.array(episode_acs)


class StirDemoEvaluation(DemoEvaluation):
    """ Evaluation for stir task. """

    def __init__(self,
                 train_id: str,
                 ckpts: List[int],
                 win_len: int = 1,
                 render_dir: str = '/tmp/resnet_train_results',
                 num_itrs: int = 50,
                 render: bool = False,
                 seed: int = 2) -> None:
        super().__init__(StirDemoGeneration, train_id, ckpts, win_len,
                         render_dir, num_itrs, render, seed)
        self.init_elem_value = 0
        self.radii = []

    def update_result(self, obs) -> None:
        grip_pos = self.Task.get_grip_pos(obs)
        center = self.Task.center
        # radius = self.Task.radius
        self.radii.append(np.linalg.norm(grip_pos - center))
        if self.timeStep >= self.Task.env._max_episode_steps * 1.2:
            self.min_dists[-1] = np.sum(np.abs(np.diff(np.array(self.radii))))
        # self.min_dists[-1] += (np.linalg.norm(grip_pos - center) - radius)**2


class PushDemoEvaluation(DemoEvaluation):
    """ Evaluation for push task. """

    def __init__(self,
                 train_id: str,
                 ckpts: List[int],
                 win_len: int = 1,
                 render_dir: str = '/tmp/resnet_train_results',
                 num_itrs: int = 50,
                 render: bool = False,
                 seed: int = 2) -> None:
        super().__init__(PushDemoGeneration, train_id, ckpts, win_len,
                         render_dir, num_itrs, render, seed)

    def update_result(self, obs) -> None:
        object_pos = self.Task.get_object_pos(obs)
        goal = self.Task.get_goal(obs)
        if np.linalg.norm(goal - object_pos) < self.min_dists[-1]:
            self.min_dists[-1] = np.linalg.norm(goal - object_pos)


if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()
    train_ids = ['0209_01', '0209_02', '0209_03']
    ckpts = [10, 80, 150]
    # ckpts = [10, 20, 40, 60, 80, 100, 150, 200]
    results = {}
    reach_eval = DemoEvaluation(ReachDemoGeneration,
                                train_ids[0],
                                ckpts,
                                render=False)
    # for train_id in train_ids:
    #     reach_eval.train_id = train_id
    #     results[train_id] = deepcopy(reach_eval.evaluate())

    # # averages = {
    # #     id: {ckpt: np.mean(dists) for ckpt, dists in value.items()}
    # #     for id, value in results.items()
    # # }

    # averages = {
    #     ckpt: np.concatenate([results[id][ckpt] for id in results.keys()])
    #     for ckpt in ckpts
    # }

    # # global_averages = [
    # #     (np.mean([averages[id][ckpt] for id in averages]), ckpt) for ckpt in ckpts
    # # ]

    # global_averages = [
    #     (np.mean(dists), ckpt)
    #     for ckpt, dists in averages.items()
    # ]
    # global_averages.sort()

    # min_avg_dist, best_ckpt = global_averages[0]

    # thresholds = np.arange(0.01, 0.16, 0.01)
    # success_rates = []

    # for thresh in thresholds:
    #     success_rates.append(
    #         np.sum(averages[best_ckpt] <= thresh) / len(averages[best_ckpt])
    #     )

    # print()
    # push_eval = PushDemoEvaluation(train_id, ckpts, render=True)
    # stir_eval = StirDemoEvaluation(train_id, ckpts, render=True)
    reach_eval.evaluate()
    # push_eval.evaluate()
    # stir_eval.evaluate()