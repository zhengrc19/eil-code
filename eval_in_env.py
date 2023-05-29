from typing import Any, Dict, List, Tuple
from imitation_learning.DemoEvaluation import \
    DemoEvaluation, StirDemoEvaluation, PushDemoEvaluation
from demo_generation.DemoGeneration import ReachDemoGeneration
from copy import deepcopy
from openpyxl import load_workbook
import numpy as np
import os.path as osp
import argparse

DEFAULT_CKPTS = [10, 20, 40, 60, 80, 100, 150, 200, "min"]
# DEFAULT_CKPTS = ["min"]


def main(train_ids: List[str],
         dataset: str,
         overwrite: bool = False,
         ckpts: List[int] = DEFAULT_CKPTS):
    results, used_state = generate_eval_results(train_ids, dataset, overwrite,
                                                ckpts)
    final_result = analyze_results(results, train_ids, dataset, used_state,
                                   ckpts)
    add_to_workbook(res_to_row(final_result))


def generate_eval_results(
    train_ids: List[str],
    dataset: str,
    overwrite: bool = False,
    ckpts: List[int] = DEFAULT_CKPTS
) -> Tuple[Dict[str, dict], bool, DemoEvaluation]:
    filename = osp.join(
        "/tmp/resnet_train_results/",
        f"{train_ids[0]}_{'' if int(train_ids[0][-1]) < 3 else 'no'}state.npz")

    if not overwrite and osp.exists(filename):
        print(f"Loading results from {filename}.")
        results = np.load(filename, allow_pickle=True)
        return results, int(train_ids[0][-1]) < 3, None

    results = {}
    if "reach" in dataset:
        evaluator = DemoEvaluation(ReachDemoGeneration,
                                   train_ids[0],
                                   ckpts,
                                   render=True)
    elif "push" in dataset:
        evaluator = PushDemoEvaluation(train_ids[0], ckpts, render=True)
    elif "stir" in dataset:
        evaluator = StirDemoEvaluation(train_ids[0], ckpts, render=True)
    else:
        raise ValueError("--dataset only supports reach, push, and stir.")

    used_state = evaluator.use_state

    for train_id in train_ids:
        evaluator.train_id = train_id
        results[train_id] = deepcopy(evaluator.evaluate())

    np.savez_compressed(filename, **results)

    return results, used_state


def analyze_results(results: Dict[str, dict],
                    train_ids: List[str],
                    dataset: str,
                    used_state: bool,
                    ckpts: List[int] = DEFAULT_CKPTS) -> Dict[str, Any]:
    try:
        all_dists = {
            ckpt: np.concatenate([results[id][ckpt] for id in results.keys()])
            for ckpt in ckpts
        }
    except:
        all_dists = {
            ckpt:
            np.concatenate([results[id].item()[ckpt] for id in results.keys()])
            for ckpt in ckpts
        }

    global_averages = [(np.mean(dists), ckpt)
                       for ckpt, dists in all_dists.items()]

    global_averages.sort()

    min_dist_avg, best_ckpt = global_averages[0]
    min_dist_std = np.std(all_dists[best_ckpt], ddof=1)

    thresholds = np.arange(0.01, 0.16, 0.01)
    success_rates = np.zeros(thresholds.shape[0], dtype=float)
    n = all_dists[best_ckpt].shape[0]

    for i, thresh in enumerate(thresholds):
        success_rates[i] = np.sum(all_dists[best_ckpt] <= thresh) / n

    rate_stdevs = np.sqrt(n * success_rates * (1 - success_rates)) / n

    final_result = {
        "DATASET": dataset,
        "USE_STATE": used_state,
        "TRAIN_IDS": train_ids,
        "BEST_EPOCH": best_ckpt,
        "MEAN_MIN": min_dist_avg,
        "STDEV_MIN": min_dist_std,
        "SUCCESS_RATES": success_rates,
        "RATES_STDEVS": rate_stdevs,
    }

    return final_result


def res_to_row(result: Dict[str, Any]) -> list:
    return [
        result["DATASET"],
        "Yes" if result["USE_STATE"] else "No",
        str(result["TRAIN_IDS"]),
        result["BEST_EPOCH"],
        result["MEAN_MIN"],
        result["STDEV_MIN"],
    ] \
     + result["SUCCESS_RATES"].tolist() \
     + result["RATES_STDEVS"].tolist()


def add_to_workbook(row: list):
    filename = "./eval_results.xlsx"
    workbook = load_workbook(filename)
    sheet = workbook.active
    sheet.append(row)
    workbook.save(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluates result of IL in Fetch Env.')
    parser.add_argument('--train_ids',
                        nargs='+',
                        default=None,
                        required=True,
                        help="Train IDS to examine.")
    parser.add_argument('--dataset',
                        required=True,
                        default='reach_state',
                        help="Dataset used.")
    parser.add_argument('--ckpts',
                        type=int,
                        nargs='+',
                        default=DEFAULT_CKPTS,
                        help="Checkpoints to examine.")
    parser.add_argument('--overwrite',
                        dest='overwrite',
                        action='store_true',
                        help="Overwrite results even if "
                        "already generated and saved.")
    parser.set_defaults(overwrite=False)
    args = parser.parse_args()
    main(args.train_ids, args.dataset, args.overwrite, args.ckpts)