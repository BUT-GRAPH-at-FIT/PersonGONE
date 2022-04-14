from info import *
import os
from subprocess import call
import argparse


parser = argparse.ArgumentParser('Detetion, tracking, and submission creation!')
parser.add_argument('--video_id', type=str, default=None, help='Absolute path file with video id and paths')
parser.add_argument('--cuda_path', default='/usr/local/cuda-11.3/lib64', type=str, help='CUDA library path')
parser.add_argument('--submission_name', default='submission.txt', type=str, help='Output submission file')
args = parser.parse_args()

vids = load_ids_and_paths(args.video_id)

################################################################################
os.chdir('detector_and_tracker')
os.environ['LD_LIBRARY_PATH'] = args.cuda_path
os.environ['PYTHONPATH'] = os.getcwd()
for vid in vids:
    call(['python', 'tools/detector_with_tracker.py',
           '-expn', 'yolox_l',
           '--path', os.path.join(inpainting_path, vid['name'], vid['name']+'.mp4'),
           '--roi_path', os.path.join(rois_path, vid['name']+'.json'),
           '--tracker', 'BYTE',
           '--tsize', str(800),
           '-f', 'exps/aic_yolox_l.py',
           '-c', 'checkpoints/yolox_l/best_ckpt.pth'
           ])
################################################################################

################################################################################
os.chdir('..')
call(['python', 'utils/tracks_processing.py',
      os.path.abspath(os.path.join('detector_and_tracker', 'YOLOX_outputs', 'yolox_l')),
      args.submission_name,
      args.video_id
      ])
################################################################################
