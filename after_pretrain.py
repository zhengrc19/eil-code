from os import environ
import random
import subprocess
from glob import glob
import os.path as osp
import sys


def main():
    eil_train_id, ckpt_idx, dataset, gpu  = sys.argv[1:5]
    try:
        ckpt_idx = int(ckpt_idx)
    except:
        raise ValueError("sys.argv[2] needs to be an int.")

    # dataset = "reach_state"
    il_dataset = f"{dataset}_{eil_train_id}"
    save_dir = osp.abspath("./chosen_embs")
    LOG_ROOT = "/tmp/eil/runs/"
    DATA_ROOT = "/tmp/"
    DATA_PREFIX = "xirl_format_datasets/train/"
    CWD = osp.dirname(osp.abspath(__file__))
    ENV = dict(environ, CUDA_VISIBLE_DEVICES=gpu)

    subprocess.run(
        args=[
            "python",
            "compute_embedding.py",
            "--experiment_path",
            osp.join(LOG_ROOT, f"alignment_logs_{eil_train_id}"),
            "--ckpt_idx",
            str(ckpt_idx),
            "--save_dir",
            save_dir,
            "--nofor_visualization",
        ],
        cwd=CWD,
        env=ENV
    )

    if ckpt_idx == -1:
        ckpt_filename = sorted(glob(osp.join(LOG_ROOT, f"alignment_logs_{eil_train_id}", "embeddings/*.npz")))[-1]
        ckpt_idx = int(ckpt_filename[-9:-4])

    subprocess.run(
        args=[
            "python",
            "filter_dataset.py",
            "--src_dir",
            osp.join(DATA_ROOT, DATA_PREFIX, dataset),
            "--dst_dir",
            osp.join(DATA_ROOT, DATA_PREFIX, il_dataset),
            "--embs_path",
            osp.join(save_dir, f"embeddings_{eil_train_id}_{ckpt_idx:05d}.npz"),
            "--nouse_ref",
            "--overwrite",
            # "--nohas_state",
        ],
        cwd=CWD
    )

    id_date, id_cnt = eil_train_id.split('_')
    id_cnt = int(id_cnt, base=16)
    if id_cnt >= 16:
        raise ValueError("train_id too large, count part should be less than 0x10")

    train_ids = []
    for i in range(6):
        il_train_id = f"{id_date}_{id_cnt * 0x10 + i:02x}"
        print(il_train_id)
        train_ids.append(il_train_id)
        # if i < 3:
        #     continue
        subprocess.run(
            args=[
                "python",
                "imitation_learning/imitation_learning.py",
                "--train_dataset",
                il_dataset,
                "--valid_dataset",
                dataset,
                "--train_id",
                il_train_id,
                "--use_state" if i < 3 else "--no_use_state",
                "--overwrite",
                # "--loss_type",
                # "ce"
                # "--debug"
            ],
            cwd=CWD,
            env=ENV,
        )
    
    env_gpu = gpu if int(gpu) < 8 else str(random.randint(1, 7))
    for i in range(2):
        # if i < 2:
        #     continue
        subprocess.run(
            args=[
                "python",
                "eval_in_env.py",
                "--train_ids",
                *train_ids[i * 3 : (i+1) * 3],
                "--dataset",
                f"{dataset}_{eil_train_id}",
                # "--overwrite",
            ],
            cwd=CWD,
            env=dict(environ, CUDA_VISIBLE_DEVICES=env_gpu),
        )

    print("TRAIN_ID\tDATASET_NAME\tSTATE?")
    for i in range(6):
        il_train_id = f"{id_date}_{id_cnt * 0x10 + i:02x}"
        print(il_train_id, il_dataset, "yes" if i < 3 else "no", sep='\t')


if __name__ == '__main__':
    main()