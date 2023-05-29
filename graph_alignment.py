# coding=utf-8

"""Export aligned video frames based on nearest neighbor in embedding space."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from time import time

from tqdm import tqdm
from graph_utils import align, dist_fn, get_embs_by_name, read_labels, align_no_reference

import math

from absl import app
from absl import flags
from absl import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tensorflow.compat.v2 as tf
import os
import concurrent
from glob import glob
import matplotlib.backends.backend_pdf as pdf
from PyPDF2 import PdfFileWriter, PdfFileReader

flags.DEFINE_string('embs_path', None, 'Path to embeddings. Can be regex.')
flags.DEFINE_string('embs_path_root', None, 'Path to folder of embeddings. Can be regex.')
flags.DEFINE_boolean('train_embs', False, 'whether embs are under train_embs subfolder')
flags.DEFINE_boolean('imperfect_reach', True,
                     'Whether or not dataset is imperfect reach.')
flags.DEFINE_string(
    'labels_path',
    '/home/hxu/resnet_train_demo/imperfect_demo_val/labels.npy',
    'path to labels.npy')
flags.DEFINE_boolean('use_dtw', False, 'Use dynamic time warping.')
flags.DEFINE_integer('reference_video', 0, 'Reference video.')
flags.DEFINE_boolean('normalize_embeddings', False,
                     'If True, L2 normalizes the embeddings before aligning.')
flags.DEFINE_boolean('use_ref', True, 'Use reference video to align.')
flags.DEFINE_boolean('use_org', False, 'Use fake reference video')
flags.DEFINE_boolean('use_med', False, 'Use median to get reference video')
flags.DEFINE_boolean('use_voted', False, 'Use voted frames directly')
flags.DEFINE_boolean('use_weight', False, 'Use voted to get weighted average')
flags.DEFINE_boolean('use_mask', False, 'do not use last frame to average')
flags.DEFINE_integer('align_tolerance', 2, 'tolarance for align without reference')
# flags.mark_flag_as_required('embs_path')

gfile = tf.io.gfile
EPSILON = 1e-7
FLAGS = flags.FLAGS


def graph_alignment(embs, names, nns, title, use_dtw, query,
                    imperfect_reach, labels_path):
    """Create aligned frames."""
    if len(embs) >=36:
        embs = embs[:36]
        names = names[:36]
        nns = nns[:36]
    if imperfect_reach:
        labels, is_deviation, cumm_lens = read_labels(labels_path)

    ncols = int(math.sqrt(len(embs)))
    # nns = []
    # for candidate in range(len(embs)):
    #     nns.append(align(embs[query], embs[candidate], use_dtw))
    # # ims = []

    fig, ax = plt.subplots(
        ncols=ncols,
        nrows=ncols,
        figsize=(10 * ncols, 10 * ncols),
        tight_layout=False)
    plt.suptitle('Alignment ' + title, fontsize=72)
    for i in range(len(embs)):
        plt.sca(ax[i % ncols, i // ncols])
        vid_num = names[i]
        plt.title('Video ' + vid_num)
        vid_num = int(vid_num)
        plt.xticks(range(0, len(embs[query])))
        plt.xlabel('Demo')
        # plt.yticks(range(0, len(embs[query])))
        plt.yticks(range(0, len(embs[i])))
        plt.ylabel('Matched')
        # plt.xticks(range(0, len(embs[i])))
        if imperfect_reach:
            for j in range(len(embs[i])):
                if not vid_num:
                    color = 'k'
                else:
                    idx = cumm_lens[vid_num-1] + j
                    color = 'r' if is_deviation[int(idx)] else 'k'
                ax[i % ncols, i // ncols].get_yticklabels()[j].set_color(color)
                # ax[i%3, i//3].get_xticklabels()[j].set_color(color)
    #             print(ax[i%3, i//3].get_yticklines())
                ax[i % ncols, i // ncols].get_yticklines()[j].set_color(color)
                # ax[i%3, i//3].get_xticklines()[j].set_color(color)
    for i in range(len(embs)):
        axis = ax[i % ncols][i // ncols]
        vid_num = int(names[i])

        axis.grid()
        axis.set_aspect("equal")
        axis.tick_params(axis='x', color='black',
                         which='major', length=7, labelrotation=90)
        # axis.tick_params(axis='y', color = 'black',
        # which='major', length=7, labelrotation = 0)
        if imperfect_reach:
            symbols = "ooooo"  # "s^<>v"
            alphas = [1.0, 0.8, 0.6, 0.4, 0.2]
            for j in range(len(embs[query])):
                if not vid_num: # reference video itself must be black
                    color = 'k'
                else:
                    idx = cumm_lens[vid_num-1] + nns[i][j]
                    color = 'r' if is_deviation[int(idx)] else 'k'
                dists = np.array([
                    dist_fn(embs[query][j], embs[cand][nns[cand][j]])
                    for cand in range(len(embs))
                ])
                frame_dists = np.sqrt(np.sum(
                    (embs[query][j] - embs[i]) ** 2,
                    axis=1
                ))
                frame_rank = np.argsort(frame_dists)
                for k in range(1, 5):
                    if not i:
                        point_color = 'k'
                    else:
                        idx = cumm_lens[vid_num - 1] + frame_rank[k]
                        point_color = 'r' if is_deviation[int(idx)] else 'k'
                    axis.plot(
                        j, frame_rank[k], point_color + symbols[k], alpha=alphas[k])
                rank = np.argsort(dists)
                top5 = np.zeros(dists.shape, dtype=bool)
                top5[rank[:5]] = True
                marker = 'o' if top5[i] else 'x' # mark o if matched frame is the top-5 closest, else x
                axis.plot(j, nns[i][j], color + marker)
                # axis.plot(nns[i][j], j, color + 'o')
            rank_labels = ['1st', '2nd', '3rd', '4th', '5th']
            axis.legend(handles=[
                Line2D([0], [0], marker=symbols[idx], color='w',
                       label=rank_labels[idx], markerfacecolor='k',
                       markersize=10, alpha=alphas[idx])
                for idx in range(len(symbols))
            ])
        axis.plot(range(len(embs[query])), nns[i], '-')
        # axis.plot(nns[i], range(len(embs[query])), '-')
    return fig


def graph_alignment_no_ref(embs, names, nns, title, labels_path):
    """Create aligned frames."""
    if len(embs) >=36:
        embs = embs[:35]
        names = names[:35]
        nns = nns[:35]
    ncols = math.ceil(math.sqrt(len(nns)))
    labels, is_deviation, cumm_lens = read_labels(labels_path)

    fig, ax = plt.subplots(
        ncols=ncols,
        nrows=ncols,
        figsize=(10 * ncols, 10 * ncols),
        tight_layout=False)
    plt.suptitle('Alignment ' + title, fontsize=72)
    for i in range(len(embs)):
        plt.sca(ax[(i + 1) % ncols, (i + 1) // ncols])
        plt.title('Video ' + names[i])
        vid_num = int(names[i])
        plt.xticks(range(0, len(nns[i])))
        plt.xlabel('Demo')
        plt.yticks(range(0, len(embs[i])))
        plt.ylabel('Matched')
        # plt.xticks(range(0, len(embs[i])))
        for j in range(len(embs[i])):
            idx = cumm_lens[vid_num-1] + j
            color = 'r' if is_deviation[int(idx)] else 'k'
            ax[(i + 1) % ncols, (i + 1) // ncols].get_yticklabels()[j].set_color(color)
            ax[(i + 1) % ncols, (i + 1) // ncols].get_yticklines()[j].set_color(color)
    for i in range(len(nns)):
        axis = ax[(i + 1) % ncols, (i + 1) // ncols]
        axis.grid()
        axis.set_aspect("equal")
        axis.tick_params(axis='x', color='black', which='major', length=7, labelrotation=90)
        vid_num = int(names[i])
        for j in range(len(nns[i])):
            idx = cumm_lens[vid_num-1] + nns[i][j]
            color = 'r' if is_deviation[int(idx)] else 'k'
            marker = 'o'
            axis.plot(j, nns[i][j], color + marker)

        axis.plot(range(len(nns[i])), nns[i], '-')
    return fig

def graph():
    if FLAGS.embs_path_root: # plot all figures in one pdf file
        if 'alignment' in FLAGS.embs_path_root:
            s = FLAGS.embs_path_root[FLAGS.embs_path_root.find('alignment')+15:][:7]
        else:
            s = os.getcwd()[os.getcwd().find('alignment') + 15:][:7]
        if FLAGS.train_embs:
            s = s.split('/')[0]
        s += "_dtw" if FLAGS.use_dtw else "_nodtw"
        s += "_noref" if not FLAGS.use_ref else ""
        s += "_tolerance_%d" % FLAGS.align_tolerance if not FLAGS.use_ref else ""
        s += "_org" if FLAGS.use_org else ""
        s += "_med" if FLAGS.use_med else ""
        s += "_vote" if FLAGS.use_voted else ""
        s += "_weight" if FLAGS.use_weight else ""
        s += "_mask" if FLAGS.use_mask else ""
        file_name, bookmarks = 'align_curve_' + s + '.pdf', []
        with pdf.PdfPages(file_name) as pdf_file:
            embs_paths = glob(os.path.join(FLAGS.embs_path_root, '*.np[yz]'))
            embs_paths = sorted(embs_paths,key=lambda x: x.rsplit('_', 2)[1] if FLAGS.train_embs else x.rsplit('_', 1)[1])
            for embs_path in embs_paths:
                s = embs_path.split('.')[-2]
                s = s[s.find('embeddings') + 11:]
                # embs, names, nns = get_embs_by_name(
                #     embs_path, FLAGS.reference_video, FLAGS.use_dtw)
                # if FLAGS.use_ref:
                #     fig = graph_alignment(embs, names, nns, s, FLAGS.use_dtw, query=FLAGS.reference_video, 
                #         imperfect_reach=FLAGS.imperfect_reach, labels_path=FLAGS.labels_path)
                # elif FLAGS.use_org:
                #     nns = []
                #     nns_no_ref, embs_no_ref, confs = align_no_reference(embs[1:], FLAGS.use_org, FLAGS.use_dtw, False, FLAGS.use_med, FLAGS.use_weight, FLAGS.use_mask, FLAGS.align_tolerance)
                #     # print(confs)
                #     for candidate in range(len(embs)):
                #         nns.append(align(embs_no_ref, embs[candidate], FLAGS.use_dtw))
                #     embs[FLAGS.reference_video] = embs_no_ref
                #     fig = graph_alignment(embs, names, nns, s, FLAGS.use_dtw, query=FLAGS.reference_video, 
                #         imperfect_reach=FLAGS.imperfect_reach, labels_path=FLAGS.labels_path)
                # else:
                #     nns_no_ref, embs_no_ref, confs = align_no_reference(embs[1:], FLAGS.use_org, FLAGS.use_dtw, FLAGS.use_voted, FLAGS.use_med, FLAGS.use_weight, FLAGS.use_mask, FLAGS.align_tolerance)
                #     # print(confs) # TODO: weighted average by conf
                #     fig = graph_alignment_no_ref(embs[1:], names[1:], nns_no_ref, s, FLAGS.labels_path)
                bookmarks.append(s)
                # pdf_file.savefig(fig)
                # plt.close(fig)
            executor = concurrent.futures.ProcessPoolExecutor(20)
            futures = [executor.submit(graph_one_emb_file, embs_path) for embs_path in embs_paths]
            concurrent.futures.wait(futures)
            for fig in futures:
                pdf_file.savefig(fig.result())
                plt.close(fig.result())
        pdf_in = PdfFileReader(file_name)
        pdf_out = PdfFileWriter()
        for i in range(len(bookmarks)):
            pdf_out.addPage(pdf_in.getPage(i))
            pdf_out.addBookmark(bookmarks[i], i)
        with open(file_name, 'wb') as f:
            pdf_out.write(f)
    elif FLAGS.embs_path:
        embs, names, nns = get_embs_by_name(FLAGS.embs_path, FLAGS.reference_video, FLAGS.use_dtw)
        s = FLAGS.embs_path.split('.')[-2]
        s = s[s.find('embeddings') + 11:]
        s += "_dtw" if FLAGS.use_dtw else "_nodtw"
        s += "_noref" if not FLAGS.use_ref else ""
        s += "_tolerance_%d" % FLAGS.align_tolerance if not FLAGS.use_ref else ""
        s += "_med" if FLAGS.use_med else ""
        s += "_vote" if FLAGS.use_voted else ""
        s += "_org" if FLAGS.use_org else ""
        if FLAGS.use_org:
            raise NotImplementedError
        if FLAGS.use_ref:
            graph_alignment(embs, names, nns, s, FLAGS.use_dtw, query=FLAGS.reference_video,
                    imperfect_reach=FLAGS.imperfect_reach, labels_path=FLAGS.labels_path)
        else:
            nn_no_refs, _, _ = align_no_reference(embs[1:], FLAGS.use_org, FLAGS.use_dtw, FLAGS.use_voted, FLAGS.use_med, FLAGS.use_weight, FLAGS.align_tolerance)
            graph_alignment_no_ref(embs[1:], names[1:], nn_no_refs, s, FLAGS.labels_path)
        file_name = 'align_curve_' + s + '.pdf'
        plt.savefig(file_name)
    else:
        raise ValueError('neither embs_path nor embs_path_root is specified!')
    print('save file at', os.path.join(os.getcwd(), file_name))


def main(_):
    begin = time()
    print("Starting alignment graphing...")
    graph()
    used = time() - begin
    print(f"Finished alignment graphing. Used {int(used // 60)}m {used % 60:.2f}s.")

def graph_one_emb_file(embs_path):
    s = embs_path.split('.')[-2]
    s = s[s.find('embeddings') + 11:]
    embs, names, nns = get_embs_by_name(
        embs_path, FLAGS.reference_video, FLAGS.use_dtw)
    if FLAGS.use_ref:
        fig = graph_alignment(embs, names, nns, s, FLAGS.use_dtw, query=FLAGS.reference_video, 
            imperfect_reach=FLAGS.imperfect_reach, labels_path=FLAGS.labels_path)
    elif FLAGS.use_org:
        nns_no_ref, embs_no_ref, confs = align_no_reference(embs[1:], FLAGS.use_org, FLAGS.use_dtw, False, FLAGS.use_med, FLAGS.use_weight, FLAGS.use_mask, FLAGS.align_tolerance)
        # print(confs)
        embs[FLAGS.reference_video] = embs_no_ref
        fig = graph_alignment(embs, names, nns_no_ref, s, FLAGS.use_dtw, query=FLAGS.reference_video, 
            imperfect_reach=FLAGS.imperfect_reach, labels_path=FLAGS.labels_path)
    else:
        nns_no_ref, embs_no_ref, confs = align_no_reference(embs[1:], FLAGS.use_org, FLAGS.use_dtw, FLAGS.use_voted, FLAGS.use_med, FLAGS.use_weight, FLAGS.use_mask, FLAGS.align_tolerance)
        # print(confs) # TODO: weighted average by conf
        fig = graph_alignment_no_ref(embs[1:], names[1:], nns_no_ref, s, FLAGS.labels_path)
    # print('\r' + embs_path, end='')
    return fig

if __name__ == '__main__':
    app.run(main)
