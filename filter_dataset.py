# coding=utf-8
"""Export aligned video frames based on nearest neighbor in embedding space."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from graph_utils import get_embs_by_name, read_labels, align_no_reference
from absl import app
from absl import flags

from tqdm import trange
import os
import os.path as osp
import shutil

flags.DEFINE_string('embs_path', None, 'Path to embeddings. Can be regex.')
flags.DEFINE_string('src_dir', None, 'Dir of images to be aligned.')
flags.DEFINE_string('dst_dir', None, 'Dir of aligned images.')
flags.DEFINE_string('labels_path', None, 'Path to ground truth labels.')
flags.DEFINE_boolean('use_dtw', False, 'Use dynamic time warping.')
flags.DEFINE_boolean('use_ref', True, 'Use reference video for aligning.')
flags.DEFINE_integer('tolerance', 2, 'Use reference video for aligning.')
flags.DEFINE_boolean('use_org', False, 'Use reference video for aligning.')
flags.DEFINE_integer('reference_video', 0, 'Reference video.')
flags.DEFINE_boolean('normalize_embeddings', False,
                     'If True, L2 normalizes the embeddings before aligning.')
flags.DEFINE_boolean('overwrite', False,
                     'If True, overwrites folder even if it exists.')
flags.DEFINE_boolean('has_state', True, 'Whether or not labels.npy has state.')

flags.mark_flag_as_required('dst_dir')
flags.mark_flag_as_required('src_dir')
flags.mark_flag_as_required('embs_path')

FLAGS = flags.FLAGS


def create_frames(embs, names, nns, dst_dir, query, labels_path):
    """Create aligned frames."""

    start = 0 if FLAGS.use_ref else 1
    for i in range(start, len(embs)):
        vid_str = names[i]
        os.makedirs(osp.join(dst_dir, vid_str), exist_ok=True)
    # ncols = int(math.sqrt(len(embs)))
    nvids = len(embs)
    # nns = []
    # for candidate in range(len(embs)):
    #     nns.append(align(embs[query], embs[candidate], use_dtw))

    labels, is_deviation, cumm_lens = read_labels(labels_path)
    actions = labels['actions']
    if FLAGS.has_state:
        states = labels['states']

    tmp_is_deviation = [[] for _ in range(40)]
    tmp_actions = [[] for _ in range(40)]
    if FLAGS.has_state:
        tmp_states = [[] for _ in range(40)]
    new_seq_lens = np.zeros(40, )

    def update(i):
        # logging.info('%s/%s', i + 1, len(embs[query]))
        # for k in range(ncols):
        #     for j in range(ncols):
        for k in range(start, nvids):
            nn = nns[k][i]
            vid_str = names[k]

            vid_num = int(vid_str)
            idx = cumm_lens[vid_num - 1] + nn if vid_num else nn
            idx = int(idx)
            tmp_is_deviation[vid_num].append(is_deviation[idx])
            tmp_actions[vid_num].append(actions[idx])
            if FLAGS.has_state:
                tmp_states[vid_num].append(states[idx])
            new_seq_lens[vid_num] += 1

            nn_str = '%04d.jpg' % nn
            i_str = '%04d.jpg' % i
            src_path = osp.join(FLAGS.src_dir, vid_str, nn_str)
            dst_path = osp.join(dst_dir, vid_str, i_str)
            shutil.copyfile(src_path, dst_path)

    num_frames = len(nns[query])
    for i in trange(num_frames):
        update(i)

    if not FLAGS.use_ref:
        # transfer perfect video 0 as is
        vid0_len = int(cumm_lens[0])
        tmp_is_deviation[0] = is_deviation[:vid0_len].tolist()
        tmp_actions[0] = actions[:vid0_len].tolist()
        if FLAGS.has_state:
            tmp_states[0] = states[:vid0_len].tolist()
        new_seq_lens[0] = vid0_len
        shutil.copytree(src=osp.join(FLAGS.src_dir, '0000'),
                        dst=osp.join(dst_dir, '0000'))

    new_is_deviation = sum(tmp_is_deviation, start=[])
    new_actions = sum(tmp_actions, start=[])
    if FLAGS.has_state:
        new_states = sum(tmp_states, start=[])
    # for dev in tmp_is_deviation:
    #     new_is_deviation.extend(dev)
    # for act in tmp_actions:
    #     new_actions.extend(act)

    new_labels = {
        # 'states': np.array(new_states),
        'actions': np.array(new_actions),
        'is_deviation': np.array(new_is_deviation),
        'seq_lens': np.array(new_seq_lens)
    }
    if FLAGS.has_state:
        new_labels['states'] = np.array(new_states)

    np.save(osp.join(dst_dir, 'labels.npy'), new_labels, allow_pickle=True)


def generate():
    """Visualize alignment."""

    embs, names, nns = get_embs_by_name(FLAGS.embs_path, FLAGS.reference_video,
                                        FLAGS.use_dtw)
    query = FLAGS.reference_video
    if not FLAGS.use_ref:
        nns, ref_embs, confs = align_no_reference(embs[1:],
                                                  FLAGS.use_org,
                                                  FLAGS.use_dtw,
                                                  tolerance=FLAGS.tolerance)
        # names = names[1:]
        nns = [np.arange(len(embs[0])).tolist()] + nns.tolist()
        query = 1

    try:
        os.makedirs(FLAGS.dst_dir)
    except FileExistsError:
        if FLAGS.overwrite:
            shutil.rmtree(FLAGS.dst_dir)
            os.makedirs(FLAGS.dst_dir)
        else:
            raise FileExistsError(
                f"Folder '{FLAGS.dst_dir}' already exists. Add --overwrite to overwrite."
            )

    if FLAGS.labels_path is None:
        FLAGS.labels_path = osp.join(FLAGS.src_dir, 'labels.npy')
    create_frames(embs,
                  names,
                  nns,
                  dst_dir=FLAGS.dst_dir,
                  query=query,
                  labels_path=FLAGS.labels_path)


def main(_):
    generate()


if __name__ == '__main__':
    app.run(main)
