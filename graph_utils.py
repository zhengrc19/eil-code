import math
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
from dtw import dtw


def dist_fn(x, y):
    dist = np.sum((x-y)**2)
    return dist


def get_nn(embs, query_emb):
    dist = np.linalg.norm(embs - query_emb, axis=1)
    assert len(dist) == len(embs)
    return np.argmin(dist), np.min(dist)

def get_nn_voted(q_embs, c_embs):
    """get n frames'(from different videos) nn in candidate video and voted the most appeared, 
    use cosine distance like get_scaled_similarity in deterministic_alignment.py
    # TODO: add l2 distance ?
    Args:
        query_embs (np.ndarray, num_videos * embs_dim): embs from each query frame
        candidate_embs (np.ndarray, num_frames * embs_dim): embs for each frame in candidate
    """
    similarity = np.matmul(q_embs, c_embs.T)
    nns = similarity.argmax(axis=1)
    # print(np.bincount(nns))
    voted_nn = np.argmax(np.bincount(nns))
    votes = np.max(np.bincount(nns))
    return voted_nn, votes

def get_reference_emb(embs, use_median=False, use_weight=False, valid_mask=None):
    """get next reference embedding from unsupervised videos

    Args:
        embs (list of np.ndarray): embs for each video
        cur_frames (list of int): current matched frame for each video
    """
    ref_candidates = []
    total_votes = []
    for i in range(len(embs)):
        c_embs = embs[i]
        del embs[i] # exclude currend video and include it later
        q_embs = np.stack([emb[0] for emb in embs])
        voted, votes = get_nn_voted(q_embs, c_embs)
        total_votes.append(votes)
        ref_candidates.append(voted)
        embs.insert(i, c_embs)
    ref_embs = np.stack([embs[i][ref_candidates[i]] for i in range(len(embs))])
    if valid_mask is not None:
        ref_embs = ref_embs[valid_mask]
    confidence = round(sum(total_votes) / len(embs) / (len(embs) - 1), 2)
    # mean of ref_embs as new reference
    if use_median:
        return np.median(ref_embs, axis=0), confidence, np.array(ref_candidates)
    elif use_weight:
        return np.average(ref_embs, axis=0, weights=total_votes), confidence, np.array(ref_candidates)
    else:
        return ref_embs.mean(axis=0), confidence, np.array(ref_candidates)

def unnorm(query_frame):
    min_v = query_frame.min()
    max_v = query_frame.max()
    query_frame = (query_frame - min_v) / (max_v - min_v)
    return query_frame


def align(query_feats, candidate_feats, use_dtw):
    """Align videos based on nearest neighbor or dynamic time warping."""
    if use_dtw:
        _, _, _, path = dtw(query_feats, candidate_feats, dist=dist_fn)
        _, uix = np.unique(path[0], return_index=True)
        nns = path[1][uix]
    else:
        nns = []
        for i in range(len(query_feats)):
            nn_frame_id, _ = get_nn(candidate_feats, query_feats[i])
            nns.append(nn_frame_id)
    return nns

def align_reference(embs, ref_emb, use_dtw=False):
    nns = []
    for candidate in range(len(embs)):
        nns.append(align(ref_emb, embs[candidate], use_dtw))
    return nns

def align_no_reference(embs, use_org=True, use_dtw=False, use_voted=False, use_median=False, use_weight=False, use_mask=False, tolerance=2):
    """align n videos without a reference video
    Args:
        embs (list of np.ndarray): embs of each video, reference video shouldn't be included
        selected_frames, reference_embs = align_no_reference(embs[1:])
    """
    assert not (use_org and use_voted)
    assert tolerance > 1
    num_videos = len(embs)
    video_lens = np.array([emb.shape[0] for emb in embs])
    cur_frames = np.zeros(num_videos, dtype=np.int) # use minimum to make sure cur_frame not exceeds video_lens
    voted_frames = np.zeros(num_videos, dtype=np.int)
    selected_frames = np.zeros(num_videos) # placeholder
    reference_embs, embs_confs = [], []
    while not (cur_frames==video_lens - 1).all():
        embs_clipped = [embs[i][cur_frames[i]:, :] for i in range(len(embs))] # begin align from current frame on
        valid_mask = (cur_frames != video_lens - 1) if use_mask else None
        cur_ref, conf, cur_voted = get_reference_emb(embs_clipped, use_median, use_weight, valid_mask)
        embs_confs.append(conf)
        reference_embs.append(cur_ref)
        for i, emb in enumerate(embs_clipped):
            sim_to_ref = np.matmul(cur_ref, emb.T)[1:]
            if len(sim_to_ref) >= tolerance:
                cur_frames[i] += sorted(np.argsort(sim_to_ref)[-tolerance:])[0] + 1
            else:
                cur_frames[i] += 1
            # candidate frame when cur_frame doesn't change
            # if len(sim_to_ref) >= tolerance:
            #     candidate_frames = sorted(np.argsort(sim_to_ref)[-tolerance:])
            # else:
            #     candidate_frames = sorted(np.argsort(sim_to_ref)) + [0]
            # cur_frames_candidate[i] = cur_frames[i] + candidate_frames[1] + 1
            # cur_frames[i] += candidate_frames[0] + 1
        voted_frames += cur_voted + 1
        voted_frames = np.minimum(voted_frames, video_lens - 1)
        cur_frames = np.minimum(cur_frames, video_lens - 1)
        # if np.equal(cur_frames, cur_frames_last).all():
        #     # cur_frames = np.minimum(cur_frames + 1, video_lens - 1)
        #     cur_frames = np.minimum(cur_frames_candidate, video_lens - 1)
        # cur_frames_last = cur_frames
        selected_frames = np.c_[selected_frames, voted_frames[:,None]] if use_voted else np.c_[selected_frames, cur_frames[:,None]]
    reference_embs = np.stack(reference_embs)
    selected_frames = selected_frames.astype(np.int32)
    if use_org:
        selected_frames = []
        for candidate in range(len(embs)):
            selected_frames.append(align(reference_embs, embs[candidate], use_dtw))
    return selected_frames, reference_embs, embs_confs

def align_no_reference_drop(embs):
    # align unsupervised video by drop current frames
    num_videos = len(embs)
    video_lens = np.array([emb.shape[0] for emb in embs])
    cur_frames = np.zeros(num_videos, dtype=np.int) # use minimum to make sure cur_frame not exceeds video_lens
    cur_frames_last = np.zeros(num_videos, dtype=np.int)
    selected_frames = np.zeros((num_videos, 1)) # placeholder
    real_frames = np.zeros(num_videos, 1)
    reference_embs = []
    while not (cur_frames==video_lens - 1).all():
        cur_embs = [emb[cur_frames[i]] for i, emb in enumerate(embs)]
        cur_ref, conf, ref_frames = get_reference_emb(cur_embs)
        reference_embs.append(cur_ref)
        for i in range(len(embs)):
            embs[i] = np.delete(embs[i], cur_ref[i], 0)
            sim_to_ref = np.matmul(cur_ref, embs[i].T)
            cur_frames[i] = np.argmax(sim_to_ref)
        video_lens -= 1
        cur_frames = np.minimum(cur_frames, video_lens - 1)
        real_frames = np.c_[real_frames, (selected_frames <= cur_frames[:, None]).sum(axis=1, keepdims=True)]
        selected_frames = np.c_[selected_frames, cur_frames[:,None]]
    selected_frames = selected_frames[:, 1:]
    reference_embs = np.stack(reference_embs)
    return selected_frames, reference_embs
    
def get_embs_by_iter(emb_paths, query, use_dtw):
    embs_by_iter = []
    # add all iters data into embs_by_iter
    # return a list of embs
    for emb_path in emb_paths:
        if 'current' in emb_path:
            continue
        iters = int(emb_path.strip('.npy').split('_')[-1])
        query_dict = np.load(emb_path, allow_pickle=True).item()
        embs = query_dict['embs']
        query_dict['names'] = [query_dict['names'][i][0].decode()
                               for i in range(len(embs))]
        # print(query_dict['names'])
        nns = []
        for candidate in range(len(embs)):
            nns.append(align(embs[query], embs[candidate], use_dtw))
        query_dict['nns'] = nns
        embs_by_iter.append((iters, query_dict))
    return embs_by_iter

def get_embs_by_iter_no_ref(emb_paths, **align_kwargs):
# def align_no_reference(embs, use_org=True, use_dtw=False, use_voted=False, use_median=False, use_weight=False, use_mask=False, tolerance=2):

    embs_by_iter = []
    # add all iters data into embs_by_iter
    # return a list of embs
    for emb_path in emb_paths:
        if 'current' in emb_path:
            continue
        iters = int(emb_path.strip('.npy').replace('_full', '').split('_')[-1])
        if os.path.splitext(emb_path)[1] == '.npz':
            query_dict = np.load(emb_path, allow_pickle=True)
            embs = query_dict['embs']
            query_dict['names'] = [os.path.basename(query_dict['names'][i]) for i in range(len(embs))]
        else:
            embs = query_dict['embs']
            query_dict['names'] = [query_dict['names'][i][0].decode() for i in range(len(embs))]
            query_dict = np.load(emb_path, allow_pickle=True).item()
        no_ref_nns, ref_emb, _ = align_no_reference(embs[1:], use_org=False, **align_kwargs)
        query_dict['embs'][0] = ref_emb
        no_ref_nns = [np.arange(len(no_ref_nns[0]))] + list(no_ref_nns)
        query_dict['nns'] = no_ref_nns
        embs_by_iter.append((iters, query_dict))
    return embs_by_iter

def get_embs_by_name(emb_path, query, use_dtw):
    """
    get single embedding file from emb_path
    query: number of reference video for nns, default to 0
    """
    if os.path.splitext(emb_path)[1] == '.npz':
        query_dict = np.load(emb_path, allow_pickle=True)
        embs = query_dict['embs']
        names = [os.path.basename(query_dict['names'][i]) for i in range(len(embs))]
    else:
        query_dict = np.load(emb_path, allow_pickle=True).item()
        embs = query_dict['embs']
        names = [query_dict['names'][i][0].decode() for i in range(len(embs))]
    nns = []
    for candidate in range(len(embs)):
        nns.append(align(embs[query], embs[candidate], use_dtw))
    return embs, names, nns


def read_labels(labels_path):
    # print(labels_path)
    labels = np.load(labels_path, allow_pickle=True).item()
    seq_lens = labels['seq_lens']
    is_deviation = labels['is_deviation']
    cumm_lens = np.zeros(seq_lens.shape)
    cumm_sum = 0
    for i in range(seq_lens.shape[0]):
        cumm_sum += seq_lens[i]
        cumm_lens[i] = cumm_sum
    return labels, is_deviation, cumm_lens


def marginal_red_frame_ratio(query, embs_by_iter, is_deviation,
                             cumm_lens, i, vid_num):
    iters = []
    values = []
    for iter, query_dict in embs_by_iter:
        embs = query_dict['embs']
        nns = query_dict['nns']
        total_rf = 0
        margin_rf = 0
        for j in range(len(embs[query])):
            if vid_num:
                idx = int(cumm_lens[vid_num-1] + nns[i][j])
                if is_deviation[idx]:
                    total_rf += 1
                    is_margin = 0
                    if idx >= 2:
                        if not is_deviation[idx - 2]:
                            is_margin = 1
                    if idx < len(is_deviation) - 2:
                        if not is_deviation[idx + 2]:
                            is_margin = 1
                    margin_rf += is_margin
        iters.append(iter)
        values.append(margin_rf / total_rf if total_rf else 1)
    return iters, values, 0.25*np.ones(len(iters))  # baseline


def marginal_red_frame_matched(query, embs_by_iter, is_deviation,
                               cumm_lens, i, vid_num):
    iters = []
    values = []
    for iter, query_dict in embs_by_iter:
        embs = query_dict['embs']
        nns = query_dict['nns']
        total_rf = 0
        margin_rf = 0
        for j in range(len(embs[query])):
            if vid_num:
                idx = int(cumm_lens[vid_num-1] + nns[i][j])
                if is_deviation[idx]:
                    total_rf += 1
                    is_margin = 0
                    if idx >= 2:
                        if not is_deviation[idx - 2]:
                            is_margin = 1
                    if idx < len(is_deviation) - 2:
                        if not is_deviation[idx + 2]:
                            is_margin = 1
                    margin_rf += is_margin
        iters.append(iter)
        values.append(margin_rf)
    return iters, values, 4*np.ones(len(iters))  # baseline


def red_frame_fraction(query, embs_by_iter, is_deviation,
                       cumm_lens, i, vid_num):
    iters = []
    values = []
    for iter, query_dict in embs_by_iter:
        embs = query_dict['embs']
        nns = query_dict['nns']
        total_rf = 0
        for j in range(len(embs[query])):
            if vid_num:
                idx = int(cumm_lens[vid_num-1] + nns[i][j])
                total_rf += 1 if is_deviation[idx] else 0
        iters.append(iter)
        values.append(total_rf / len(embs[query]))
    return iters, values, np.ones(len(iters)) * 16 / len(embs[i])  # baseline


def red_frame_matched(query, embs_by_iter, is_deviation,
                      cumm_lens, i, vid_num):
    iters = []
    values = []
    for iter, query_dict in embs_by_iter:
        embs = query_dict['embs']
        nns = query_dict['nns']
        total_rf = 0
        for j in range(len(embs[query])):
            if vid_num:
                idx = int(cumm_lens[vid_num-1] + nns[i][j])
                total_rf += 1 if is_deviation[idx] else 0
        iters.append(iter)
        values.append(total_rf)
    return iters, values, np.ones(len(iters)) * 16  # baseline


def analyze_msd(actions, indices=None):
    # total_len = 0
    if indices is not None:
        actions = actions[indices, :3]
    else:
        actions = actions[:, :3]
    normalized = actions / \
        np.expand_dims(np.sqrt(np.sum(actions**2, axis=1)), 1)
    diffs = np.diff(normalized, axis=0)[np.mean(
        1 - np.isnan(np.diff(normalized, axis=0)), axis=1) == 1]
    return np.sum(diffs**2) / len(diffs)


def mean_square_difference(query, embs_by_iter, actions,
                           cumm_lens, i, vid_num):
    start = int(cumm_lens[vid_num - 1] if vid_num else 0)
    end = int(cumm_lens[vid_num])
    iters = []
    values = []
    for iter, query_dict in embs_by_iter:
        # embs = query_dict['embs']
        nns = query_dict['nns']
        neighbor_actions = nns[i]
        msd = analyze_msd(actions[start:end], neighbor_actions)
        iters.append(iter)
        values.append(msd)
    return iters, values, np.ones(len(iters)) * analyze_msd(actions[start:end])


def analyze_stdev(actions, indices=None):
    if indices is not None:
        actions = actions[indices, :3]
    else:
        actions = actions[:, :3]
    normalized = actions / \
        np.expand_dims(np.sqrt(np.sum(actions**2, axis=1)), 1)
    normalized = normalized[np.mean(1 - np.isnan(normalized), axis=1) == 1]
    return np.linalg.norm(normalized.std(axis=0))


def standard_dev(query, embs_by_iter, actions, cumm_lens, i, vid_num):
    start = int(cumm_lens[vid_num - 1] if vid_num else 0)
    end = int(cumm_lens[vid_num])
    iters = []
    values = []
    for iter, query_dict in embs_by_iter:
        embs = query_dict['embs']
        nns = query_dict['nns']
        neighbor_actions = nns[i]
        stdev = analyze_stdev(actions[start:end], neighbor_actions)
        iters.append(iter)
        values.append(stdev)
    baseline = np.ones(len(iters)) * analyze_stdev(actions[start:end])
    return iters, values, baseline


def plot_metric(metric_function, plt_title, plt_xlabel, plt_ylabel,
                plt_actual_label, plt_baseline_label, plt_figpath,
                embs_by_iter, query, cumm_lens, is_deviation):
    num_vids = len(embs_by_iter[0][1]['embs'])
    ncols = int(math.sqrt(num_vids))
    fig, ax = plt.subplots(
        ncols=ncols,
        nrows=ncols,
        figsize=(10 * ncols, 10 * ncols),
        tight_layout=False)

    plt.suptitle(plt_title, fontsize=72)
    for i in range(num_vids):
        axis = ax[i % ncols][i // ncols]
        plt.sca(axis)
        vid_num = embs_by_iter[0][1]['names'][i]
        plt.title('Video ' + vid_num)
        vid_num = int(vid_num)
        plt.xlabel(plt_xlabel)
        plt.ylabel(plt_ylabel)
        ##############################################
        iters, values, baseline = metric_function(
            query, embs_by_iter, is_deviation, cumm_lens, i, vid_num)
        ##############################################
        axis.grid()
        axis.tick_params(axis='x', color='black',
                         which='major', length=7, labelrotation=45)
        values = np.array(values)[np.argsort(iters)]
        iters = np.sort(iters)
        axis.plot(iters, values, 'ko-', label=plt_actual_label)
        if baseline is not None:
            axis.plot(iters, baseline, 'bo-', label=plt_baseline_label)
        axis.legend()
    plt.savefig(plt_figpath)

def make_outlier(l, s, e):
    """make video embddings with length l and outliers start from s and end at e, f[s] and f[e] included

    Args:
        s (int): start point of outliers
        e (int): end point of outliers
        l (int): total frames of the video
    """
    frames, masks = [], []
    emb_dim = 10
    outlier_factor = 3
    for i in range(l):
        if s <= i <= e:
            masks.append(True)
            frames.append(np.ones(emb_dim) * i - outlier_factor * np.ones(emb_dim))
        else:
            masks.append(False)
            frames.append(np.ones(emb_dim) * i)
    frames = np.stack(frames)
    return frames, masks

def get_test_videos():
    # video_lengths = [51, 52, 52, 54, 52, 50, 53, 51, 52, 50, 53, 54, 50, 54, 53, 52]
    # starts = [32,  4, 34,  1,  0,  7, 22, 30, 16, 23,  8, 11, 38,  6, 24, 26]
    # ends = [41, 15, 44, 11,  9, 17, 34, 43, 28, 34, 18, 21, 47, 15, 33, 35]
    video_lengths = [5,3,5,5,5]
    starts = [5,3,2,5,5]
    ends = [5,3,4,5,5]
    videos = [make_outlier(video_lengths[i], starts[i], ends[i])[0] for i in range(len(video_lengths))]
    masks = [make_outlier(video_lengths[i], starts[i], ends[i])[1] for i in range(len(video_lengths))]
    arr = np.zeros((len(video_lengths), max(video_lengths)), bool)
    pad_mask = np.arange(max(video_lengths)) >= np.array(video_lengths)[:,None]
    arr[~pad_mask] = np.concatenate(masks)
    true_mask = np.ma.array(arr, mask=[pad_mask])
    return videos, true_mask