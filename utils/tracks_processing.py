# BEWARE: class_id is increased +1 to challenge report (not list index)
import pickle
import json
import numpy as np
import math
import os
import sys

def load_ids_and_paths(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    vids = [d.strip().split(' ') for d in data]
    pth = os.path.dirname(os.path.abspath(file_path))
    vids = [{'id' : v[0], 'vid_path' : os.path.join(pth, v[1]), 'name' : v[1].split('.')[0]} for v in vids]
    return vids

vids = load_ids_and_paths(sys.argv[3])

def get_mid_point(detection):
    return (int((detection[0]+detection[2])/2), int((detection[1]+detection[3])/2))

def merge_tracks_by_ID(detection_track_data):
    # 'merge' by track ID
    tracks = {}
    for frame_num in detection_track_data:
        for det in detection_track_data[frame_num]:
            if det['track_id'] not in tracks:
                tracks[det['track_id']] = []
            tracks[det['track_id']].append({'pos' : get_mid_point(det['det']),
                                            # 'det' : det['det'],
                                            'frame' : frame_num,
                                            'class' : int(det['class']),
                                            'class_conf' : det['class_conf']
                                            })
    return tracks


def filter_tracks_by_ROI(tracks, roi):
    # 'filter' by ROI position
    to_remove = []

    for t in tracks:
        keep = False
        for det in tracks[t]:
            if roi[0] < det['pos'][0] < roi[2] and roi[1] < det['pos'][1] < roi[3]:
                keep = True
        if not keep:
            to_remove.append(t)

    for t in to_remove:
        del(tracks[t])

    return tracks


def get_track_classes(tracks):
    # compute track class by mean of single class confidences (weighted)
    tracks_classes = {}

    for t in tracks:
        tracks_classes[t] = {'class' : None, 'dets' : [], 'probs_of_classes' : {}}
        for det in tracks[t]:
            tracks_classes[t]['dets'].append({'class' : det['class'], 'class_conf' : det['class_conf']})

    for t in tracks_classes:
        # insert probability for eaech detection (by class)
        for det in tracks_classes[t]['dets']:
            if det['class'] not in tracks_classes[t]['probs_of_classes']:
                tracks_classes[t]['probs_of_classes'][det['class']] = {'confs' : []}
            tracks_classes[t]['probs_of_classes'][det['class']]['confs'].append(det['class_conf'])

        # compute count of detections for each class and mean confidence value
        for class_id in tracks_classes[t]['probs_of_classes']:
            tracks_classes[t]['probs_of_classes'][class_id]['dets_count'] = len(tracks_classes[t]['probs_of_classes'][class_id]['confs'])
            tracks_classes[t]['probs_of_classes'][class_id]['mean'] = np.mean(tracks_classes[t]['probs_of_classes'][class_id]['confs'])

        # count of all class detections for track
        all_track_dets_cnt = 0
        for class_id in tracks_classes[t]['probs_of_classes']:
            all_track_dets_cnt += tracks_classes[t]['probs_of_classes'][class_id]['dets_count']

        # weighted mean of single classes (mean confidence value weighted by count of detections)
        for class_id in tracks_classes[t]['probs_of_classes']:
            tracks_classes[t]['probs_of_classes'][class_id]['weighted_mean'] = \
                    tracks_classes[t]['probs_of_classes'][class_id]['mean'] * \
                    (tracks_classes[t]['probs_of_classes'][class_id]['dets_count']/all_track_dets_cnt)

        # locate class with the highest weighted score
        best_class = None
        best_class_value = 0.0
        for class_id in tracks_classes[t]['probs_of_classes']:
            if tracks_classes[t]['probs_of_classes'][class_id]['weighted_mean'] > best_class_value:
                best_class_value = tracks_classes[t]['probs_of_classes'][class_id]['weighted_mean']
                best_class = class_id
        tracks_classes[t]['class'] = best_class

    return tracks_classes


def merge_by_position_and_class(tracks, tracks_classes):
    # merge close detections with came class (in some spatial and time 'window')
    time_frame = 90
    position_frame = 100

    to_remove = []
    for pos_1, t_1 in enumerate(tracks):
        for pos_2, t_2 in enumerate(tracks):
            if pos_2 <= pos_1:
                continue
            dist = np.linalg.norm(np.array(tracks[t_1][-1]['pos']) - np.array(tracks[t_2][0]['pos']))
            # if dist < position_frame and tracks_classes[t_1]['class'] == tracks_classes[t_2]['class']:
            if dist < position_frame and \
             tracks_classes[t_1]['class'] == tracks_classes[t_2]['class'] and \
             np.abs(tracks[t_1][-1]['frame'] - tracks[t_2][0]['frame']) < time_frame:
                tracks[t_1] += tracks[t_2]
                tracks_classes[t_1] = {**tracks_classes[t_1], **tracks_classes[t_2]}
                to_remove.append(t_2)

    for t in to_remove:
        try:
            del(tracks[t])
            del(tracks_classes[t])
        except:
            pass




def export_submission(tracks, tracks_classes, out_file, scene_num):
    '''
        BEWARE: class number + 1 (to evaluation)
    '''
    tracks_time = {}
    # store frame numbers for all track detections
    for t in tracks:
        tracks_time[t] = {'class' : tracks_classes[t]['class'], 'frames_nums' : []}
        for det in tracks[t]:
            tracks_time[t]['frames_nums'].append(det['frame'])

    # compute time of track detection
    for t in tracks:
        secs = [math.floor(frame/60.0) for frame in tracks_time[t]['frames_nums']]
        txt = '{:d} {:d} {:d}'.format(scene_num, tracks_time[t]['class']+1, np.argmax(np.bincount(secs)))
        out_file.write(txt+'\n')



roi_expand = 0.1


scenes = sorted(os.listdir(sys.argv[1]))
scenes = [s.split('.')[0] for s in scenes]

out_file = open(sys.argv[2], 'w')

for pos, s in enumerate(scenes):
    pth = os.path.join(sys.argv[1], s+'.pkl')

    with open(pth, 'rb') as f:
        detection_track_data = pickle.load(f)

    # with open('../data/ROIs/{:s}.json'.format(s), 'r') as f:
    #     roi = json.load(f)
    #
    # roi_w, roi_h = roi[2] - roi[0], roi[3] - roi[1]
    #
    # roi = [int(roi[0]-roi_w*(roi_expand/2)),
    #        int(roi[1]-roi_h*(roi_expand/2)),
    #        int(roi[2]+roi_w*(roi_expand/2)),
    #        int(roi[3]+roi_h*(roi_expand/2))]

    # print(pth)
    tracks = merge_tracks_by_ID(detection_track_data)
    print('Total tracks before filtering:', len(tracks))

    ############################################################
    # tracks = filter_tracks_by_ROI(tracks, roi)
    ############################################################

    # print(list(tracks.keys()))
    print('Tracks after position filtering:', len(tracks))
    tracks_classes = get_track_classes(tracks)

    merge_by_position_and_class(tracks, tracks_classes)

    print('Tracks after merging by position and class:', len(tracks))

    scene_num = 0
    for v in vids:
        if v['name'] == s:
            scene_num = v['id']
            break
    export_submission(tracks, tracks_classes, out_file, int(scene_num))

out_file.close()
