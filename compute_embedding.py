"""Compute and store the mean goal embedding using a trained model."""

import os
import os.path as osp

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
from torchkit import CheckpointManager
import utils
from xirl import common

# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_path", None, "Path to model checkpoint.")
flags.DEFINE_string("save_dir", "./chosen_embs",
                    "Directory to save output embeddings.")
flags.DEFINE_boolean(
    "restore_checkpoint", True,
    "Restore model checkpoint. Disabling loading a checkpoint is useful if you "
    "want to measure performance at random initialization.")
flags.DEFINE_integer("ckpt_idx", -1,
                     "Checkpoint index to restore. -1 means last checkpoint.")
flags.DEFINE_boolean("for_visualization", True, "16 videos or full dataset.")


def setup():
    """Load the latest embedder checkpoint and dataloaders."""
    config = utils.load_config_from_dir(FLAGS.experiment_path)
    model = common.get_model(config)
    loader = common.get_downstream_dataloaders(config, debug=True)["train"]
    checkpoint_dir = osp.join(FLAGS.experiment_path, "checkpoints")
    global_step = 0
    if FLAGS.restore_checkpoint:
        ckpt_manager = CheckpointManager(checkpoint_dir, model=model)
        global_step = utils.restore_ckpt(ckpt_manager, FLAGS.ckpt_idx)
    else:
        logging.info("Skipping checkpoint restore.")
    return model, loader, global_step


def main(_):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, downstream_loader, global_step = setup()
    model.to(device).eval()
    compressed = utils.embed(model=model,
                             downstream_loader=downstream_loader,
                             device=device,
                             for_visualization=FLAGS.for_visualization,
                             no_first_vid=FLAGS.for_visualization)
    train_id = osp.basename(FLAGS.experiment_path)
    if not train_id:
        train_id = osp.basename(FLAGS.experiment_path[:-1])
    if len(train_id) != 7:
        train_id = train_id[-7:]
    if not FLAGS.for_visualization:
        # sort order if not for visualization
        indices = np.argsort(compressed['names'])
        for k, v in compressed.items():
            if isinstance(v, list):
                compressed[k] = np.array(v, dtype=object)[indices].tolist()
            elif isinstance(v, np.ndarray):
                compressed[k] = v[indices]
            else:
                logging.info("Wrong data format in compressed.")
    FLAGS.save_dir = osp.abspath(FLAGS.save_dir)
    if not osp.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    utils.save_numpy(FLAGS.save_dir, compressed,
                     f"embeddings_{train_id}_{global_step:05d}.npz")


if __name__ == "__main__":
    flags.mark_flag_as_required("experiment_path")
    app.run(main)
