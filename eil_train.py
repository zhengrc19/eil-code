""" EIL training script. """

import os.path as osp
import subprocess

from absl import app
from absl import flags
from absl import logging
from configs.constants import ALGORITHMS
from torchkit.experiment import unique_id
import yaml

# pylint: disable=logging-fstring-interpolation

# Mapping from pretraining algorithm to config file.
ALGO_TO_CONFIG = {
    "eil": "configs/xmagical/pretraining/eil.py",
    "xirl": "configs/xmagical/pretraining/tcc.py",
    "lifs": "configs/xmagical/pretraining/lifs.py",
    "tcn": "configs/xmagical/pretraining/tcn.py",
    "goal_classifier": "configs/xmagical/pretraining/classifier.py",
    "raw_imagenet": "configs/xmagical/pretraining/imagenet.py",
}
# We want to pretrain on the entire demonstrations.
MAX_DEMONSTRATIONS = -1
FLAGS = flags.FLAGS

flags.DEFINE_enum("algo", None, ALGORITHMS, "The pretraining algorithm to use.")
flags.DEFINE_string("dataset", None, "Dataset to train on.")
flags.DEFINE_string("train_id", None, "Train id.")
flags.DEFINE_bool("unique_name", False,
                  "Whether to append a unique ID to the experiment name.")


def main(_):
  
  dataset = FLAGS.dataset

  # Generate a unique experiment name.
  kwargs = {
      "dataset": "fetch_env_tasks",
      "mode": "same",
      "algo": FLAGS.algo,
      "embodiment": dataset,
  }
  if FLAGS.unique_name:
      kwargs["uid"] = unique_id()
  experiment_name = FLAGS.train_id
  logging.info("Experiment name: %s", experiment_name)

  subprocess.run(
      [
          "python",
          "pretrain.py",
          "--experiment_name",
          experiment_name,
          "--raw_imagenet" if FLAGS.algo == "raw_imagenet" else "",
          "--config",
          f"{ALGO_TO_CONFIG[FLAGS.algo]}:{FLAGS.train_id}",
          "--config.data.pretrain_action_class",
          f"({repr(dataset)},)",
          "--config.data.downstream_action_class",
          f"({repr(dataset)},)",
          "--config.data.task",
          dataset,
          "--config.data.max_vids_per_class",
          f"{MAX_DEMONSTRATIONS}",
      ],
      check=True,
  )

  exp_path = osp.join("/tmp/eil/runs/", f"alignment_logs_{experiment_name}")

  subprocess.run([
      "python",
      "graph_alignment.py",
      "--embs_path_root",
      osp.join(exp_path, "embeddings"),
      "--labels_path",
      osp.join(
          "/tmp/xirl_format_datasets/train",
          dataset,
          "labels.npy"
      ),
      "--nouse_ref",
  ])

  # Dump experiment metadata as yaml file.
  with open(osp.join(exp_path, "metadata.yaml"), "w") as fp:
      yaml.dump(kwargs, fp)


if __name__ == "__main__":
  flags.mark_flag_as_required("algo")
  flags.mark_flag_as_required("train_id")
  flags.mark_flag_as_required("dataset")
  app.run(main)
