# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Useful methods shared by all scripts."""

import os
import pickle
import typing
from typing import Any, Dict, Optional

from absl import logging
import gym
from gym.wrappers import RescaleAction, FlattenObservation, TimeLimit
import matplotlib.pyplot as plt
from ml_collections import config_dict
import numpy as np
from sac import replay_buffer
from sac import wrappers
import torch
from torchkit import CheckpointManager
from torchkit.experiment import git_revision_hash
from xirl import common
from tqdm import tqdm
import xmagical
import yaml
from demo_generation.DemoGeneration import ZoomDevReachEnv, PushEnv, StirEnv
# pylint: disable=logging-fstring-interpolation
CUSTOM_ENV = {
  'reach': ZoomDevReachEnv,
  'push': PushEnv,
  'stir': StirEnv
}
ConfigDict = config_dict.ConfigDict
FrozenConfigDict = config_dict.FrozenConfigDict

# ========================================= #
# Experiment utils.
# ========================================= #


def setup_experiment(exp_dir, config, resume = False):
  """Initializes a pretraining or RL experiment."""
  #  If the experiment directory doesn't exist yet, creates it and dumps the
  # config dict as a yaml file and git hash as a text file.
  # If it exists already, raises a ValueError to prevent overwriting
  # unless resume is set to True.
  if os.path.exists(exp_dir):
    if not resume:
      raise ValueError(
          "Experiment already exists. Run with --resume to continue.")
    load_config_from_dir(exp_dir, config)
  else:
    os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
      yaml.dump(ConfigDict.to_dict(config), fp)
    with open(os.path.join(exp_dir, "git_hash.txt"), "w") as fp:
      fp.write(git_revision_hash())


def load_config_from_dir(
    exp_dir,
    config = None,
):
  """Load experiment config."""
  with open(os.path.join(exp_dir, "config.yaml"), "r") as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)
  # Inplace update the config if one is provided.
  if config is not None:
    config.update(cfg)
    return
  return ConfigDict(cfg)


def dump_config(exp_dir, config):
  """Dump config to disk."""
  # Note: No need to explicitly delete the previous config file as "w" will
  # overwrite the file if it already exists.
  with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
    yaml.dump(ConfigDict.to_dict(config), fp)


def copy_config_and_replace(
    config,
    update_dict = None,
    freeze = False,
):
  """Makes a copy of a config and optionally updates its values."""
  # Using the ConfigDict constructor leaves the `FieldReferences` untouched
  # unlike `ConfigDict.copy_and_resolve_references`.
  new_config = ConfigDict(config)
  if update_dict is not None:
    new_config.update(update_dict)
  if freeze:
    return FrozenConfigDict(new_config)
  return new_config


def load_model_checkpoint(pretrained_path, device, checkpoint=None):
  """Load a pretrained model and optionally a precomputed goal embedding."""
  config = load_config_from_dir(pretrained_path)
  model = common.get_model(config)
  model.to(device).eval()
  checkpoint_dir = os.path.join(pretrained_path, "checkpoints")
  checkpoint_manager = CheckpointManager(checkpoint_dir, model=model)
  if checkpoint is not None:
    ckpts = CheckpointManager.list_checkpoints(checkpoint_manager.directory)
    ckpt = list(filter(lambda x: str(x).split("/")[-1].split(".")[0] == checkpoint, ckpts))[0] 
    checkpoint_manager.checkpoint.restore(ckpt)
    step = int(ckpt.stem)
  else:
    step = checkpoint_manager.restore_or_initialize() #TODO: specify ckpt
  logging.info("Restored model from checkpoint %d.", step)
  return config, model


def save_pickle(experiment_path, arr, name):
  """Save an array as a pickle file."""
  filename = os.path.join(experiment_path, name)
  with open(filename, "wb") as fp:
    pickle.dump(arr, fp)
  logging.info("Saved %s to %s", name, filename)


def save_numpy(experiment_path, dic, name):
  """Save an array as a npz file."""
  filename = os.path.join(experiment_path, name)
  np.savez_compressed(filename, **dic)
  logging.info("Saved %s to %s", name, filename)


def load_numpy(embs_path, name):
  """Load a npz dictionary."""
  filename = os.path.join(embs_path, name)
  loaded = np.load(filename)
  logging.info("Successfully loaded %s from %s", name, filename)
  return loaded


def load_pickle(pretrained_path, name):
  """Load a pickled array."""
  filename = os.path.join(pretrained_path, name)
  with open(filename, "rb") as fp:
    arr = pickle.load(fp)
  logging.info("Successfully loaded %s from %s", name, filename)
  return arr


# ========================================= #
# RL utils.
# ========================================= #


def make_env(
    env_name,
    seed,
    save_dir = None,
    add_episode_monitor = True,
    action_repeat = 1,
    frame_stack = 1,
    flatten_observation = False,
    max_episode_steps = None
):
  """Env factory with wrapping.

  Args:
    env_name: The name of the environment.
    seed: The RNG seed.
    save_dir: Specifiy a save directory to wrap with `VideoRecorder`.
    add_episode_monitor: Set to True to wrap with `EpisodeMonitor`.
    action_repeat: A value > 1 will wrap with `ActionRepeat`.
    frame_stack: A value > 1 will wrap with `FrameStack`.

  Returns:
    gym.Env object.
  """
  # Check if the env is in x-magical.
  xmagical.register_envs()
  if env_name in CUSTOM_ENV.keys():
    env = CUSTOM_ENV[env_name]()
  elif env_name == 'gridworld':
    env = MediumMazeEnv(flatten_obs=True)
  elif env_name in xmagical.ALL_REGISTERED_ENVS:
    env = gym.make(env_name)
  else:
    raise ValueError(f"{env_name} is not a valid environment name.")

  if flatten_observation and env_name in CUSTOM_ENV:
    env = FlattenObservation(env)
  if max_episode_steps is not None:
    env = TimeLimit(env, max_episode_steps)
  if add_episode_monitor:
    env = wrappers.EpisodeMonitor(env)
  if action_repeat > 1:
    env = wrappers.ActionRepeat(env, action_repeat)
  if type(env.action_space) is gym.spaces.box.Box:
    env = RescaleAction(env, -1.0, 1.0)
  if save_dir is not None:
    env = wrappers.VideoRecorder(env, save_dir=save_dir)
  if frame_stack > 1:
    env = wrappers.FrameStack(env, frame_stack)

  # Seed.
  env.seed(seed)
  env.action_space.seed(seed)
  env.observation_space.seed(seed)

  return env


def wrap_learned_reward(env, config):
  """Wrap the environment with a learned reward wrapper.

  Args:
    env: A `gym.Env` to wrap with a `LearnedVisualRewardWrapper` wrapper.
    config: RL config dict, must inherit from base config defined in
      `configs/rl_default.py`.

  Returns:
    gym.Env object.
  """
  pretrained_path = config.reward_wrapper.pretrained_path
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model_config, model = load_model_checkpoint(pretrained_path, device)

  kwargs = {
      "env": env,
      "model": model,
      "device": device,
      "res_hw": model_config.data_augmentation.image_size,
  }

  if config.reward_wrapper.type == "goal_classifier":
    env = wrappers.GoalClassifierLearnedVisualReward(**kwargs)

  elif config.reward_wrapper.type == "distance_to_goal":
    kwargs["goal_emb"] = load_pickle(pretrained_path, "goal_emb.pkl")
    kwargs["distance_scale"] = load_pickle(pretrained_path,
                                           "distance_scale.pkl")
    env = wrappers.DistanceToGoalLearnedVisualReward(**kwargs)

  else:
    raise ValueError(
        f"{config.reward_wrapper.type} is not a valid reward wrapper.")

  return env


def make_buffer(
    env,
    device,
    config,
    ckpt=None
):
  """Replay buffer factory.

  Args:
    env: A `gym.Env`.
    device: A `torch.device` object.
    config: RL config dict, must inherit from base config defined in
      `configs/rl_default.py`.

  Returns:
    ReplayBuffer.
  """

  kwargs = {
      "obs_shape": env.observation_space.shape,
      "action_shape": env.action_space.shape or (env.action_space.n,),
      "capacity": config.replay_buffer_capacity,
      "device": device,
  }

  pretrained_path = config.reward_wrapper.pretrained_path
  if not pretrained_path:
    return replay_buffer.ReplayBuffer(**kwargs)

  model_config, model = load_model_checkpoint(pretrained_path, device, ckpt)
  kwargs["model"] = model
  kwargs["res_hw"] = model_config.data_augmentation.image_size
  if model_config.frame_sampler.num_context_frames == 2:
    kwargs["stride"] = model_config.frame_sampler.context_stride
  elif (model_config.frame_sampler.num_context_frames or 1) > 2:
    raise NotImplementedError
  if config.reward_wrapper.type == "goal_classifier":
    buffer = replay_buffer.ReplayBufferGoalClassifier(**kwargs)

  elif config.reward_wrapper.type == "distance_to_goal":
    kwargs["goal_emb"] = load_pickle(pretrained_path, "goal_emb.pkl")
    kwargs["distance_scale"] = load_pickle(pretrained_path,
                                           "distance_scale.pkl")
    buffer = replay_buffer.ReplayBufferDistanceToGoal(**kwargs)

  else:
    raise ValueError(
        f"{config.reward_wrapper.type} is not a valid reward wrapper.")

  return buffer


# ========================================= #
# Misc. utils.
# ========================================= #


def plot_reward(rews):
  """Plot raw and cumulative rewards over an episode."""
  _, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
  axes[0].plot(rews)
  axes[0].set_xlabel("Timestep")
  axes[0].set_ylabel("Reward")
  axes[1].plot(np.cumsum(rews))
  axes[1].set_xlabel("Timestep")
  axes[1].set_ylabel("Cumulative Reward")
  for ax in axes:
    ax.grid(b=True, which="major", linestyle="-")
    ax.grid(b=True, which="minor", linestyle="-", alpha=0.2)
  plt.minorticks_on()
  plt.show()


def embed(
    model,
    downstream_loader,
    device,
    for_visualization=False,
    no_first_vid=False,
):
  """Embed the stored trajectories and compute embeddings."""
  embs = []
  video_lens = []
  video_idxs = []
  video_names = []
  frames = []
  feats = []
  vid_count = 0
  for class_name, class_loader in downstream_loader.items():
    # logging.info("Embedding %s.", class_name)
    for batch in tqdm(iter(class_loader), leave=False):
      name = batch['video_name']
      if no_first_vid and name[0].endswith('0000'):
        continue
      idx = batch['frame_idxs'].numpy()
      length = batch['video_len'].numpy()
      frame = batch['frames'].numpy()
      if model.use_action:
        batch['action'] = batch['action'].to(device)
      if model.use_state:
        batch['state'] = batch['state'].to(device)
      batch['frames'] = batch['frames'].to(device)
      out = model.infer(batch)
      emb = out.numpy().embs
      feat = out.numpy().feats
      video_idxs.extend(idx)
      video_lens.extend(length)
      video_names.extend(name)
      frames.extend(frame)
      embs.append(emb)
      feats.append(feat)
      vid_count += 1
      if for_visualization and vid_count == 16:
        break

  return {
    "embs":     embs,
    "seq_lens": np.array(video_lens),
    "steps":    video_idxs,
    "names":    np.array(video_names),
    "frames":   frames,
    "feats":    feats
  }


def restore_ckpt(ckpt_manager: CheckpointManager, ckpt_idx: int) -> int:
    if ckpt_idx == -1:
        global_step = ckpt_manager.restore_or_initialize()
    else:
        ckpts = CheckpointManager.list_checkpoints(ckpt_manager.directory)
        ckpt = ckpts[list(map(lambda x: x.stem,
                                ckpts)).index(str(ckpt_idx))]
        status = ckpt_manager.checkpoint.restore(ckpt)
        if not status:
            logging.info("Could not restore checkpoint index %d.",
                         ckpt_idx)
            exit()
        global_step = int(ckpt.stem)
    logging.info("Restored model from checkpoint %d.", global_step)
    return global_step