import os
import cv2
import argparse

parser = argparse.ArgumentParser('Extrack random background images from Track1 and Track3')
parser.add_argument('--t_1_path', type=str, default=None, help='Root of Track1 data', required=True)
parser.add_argument('--t_3_path', type=str, default=None, help='Root of Track3 data', required=True)
parser.add_argument('--data_path', type=str, default=None, help='Path to Data folder', required=True)

args = parser.parse_args()

os.makedirs(os.path.join(args.data_path, 'bckg_images'), exist_ok=True)

counter = 0

################################################################################

# path_1 = '/mnt/matylda0/ispanhel/Datasets/AIC2022/AIC22_Track1_MTMT_Tracking'
path_1 = args.t_1_path

# types = ['test', 'train', 'validation']
types = ['train', 'validation']

for t in types:
    scenes = sorted(os.listdir(os.path.join(path_1, t)))
    for s in scenes:
        cams = sorted(os.listdir(os.path.join(path_1, t, s)))
        for c in cams:
            pth = os.path.join(path_1, t, s, c, 'vdo.avi')
            cap = cv2.VideoCapture(pth)
            print('Processing:', pth)

            i = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if i%30 == 0:
                    cv2.imwrite(os.path.join(args.data_path, 'bckg_images', '{:08d}.jpg'.format(counter)), frame)
                    counter += 1
                i += 1

################################################################################

################################################################################

# path_3 = '/mnt/matylda0/ispanhel/Datasets/AIC2022/AIC22_Track3_ActionRecognition'
path_3 = args.t_3_path

types = ['A1', 'A2']

for t in types:
    users = sorted(os.listdir(os.path.join(path_3, t)))
    for u in users:
        cams = sorted(os.listdir(os.path.join(path_3, t, u)))
        cams = [c for c in cams if 'MP4' in c]
        for c in cams:
            pth = os.path.join(path_3, t, u, c)
            cap = cv2.VideoCapture(pth)
            print(pth)

            i = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if i%90 == 0:
                    cv2.imwrite(os.path.join(args.data_path, 'bckg_images', '{:08d}.jpg'.format(counter)), frame)
                    counter += 1
                i += 1

################################################################################
