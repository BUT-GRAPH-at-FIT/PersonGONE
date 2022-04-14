import os
from subprocess import call
import argparse
import shutil

parser = argparse.ArgumentParser('Train YOLOX-L detector on generated dataset!')
parser.add_argument('--dataset_path', type=str, default=None, help='Absolute path to the dataset')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--device', default=None, type=int, help='device for training')
parser.add_argument('--cuda_path', default='/usr/local/cuda-11.3/lib64', type=str, help='device for training')
args = parser.parse_args()

abs_dataset_path = os.path.abspath(args.dataset_path)

os.environ['LD_LIBRARY_PATH'] = args.cuda_path
os.chdir('detector_and_tracker')
os.environ['PYTHONPATH'] = os.getcwd()

call(['python', 'tools/train.py',
      '-expn', 'yolox_l',
      '--dataset_path', abs_dataset_path,
      '-b', str(args.batch_size),
      '-d', str(args.device),
      '-f', 'exps/aic_yolox_l.py',
      '--cache'
      ])

shutil.move(os.path.join('YOLOX_outputs', 'yolox_l', 'best_ckpt.pth'), os.path.join('checkpoints', 'yolox_l', 'best_ckpt.pth'))
shutil.rmtree(os.path.join('YOLOX_outputs', 'yolox_l'))
