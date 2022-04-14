import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import json
import argparse


parser = argparse.ArgumentParser('Generate training dataset in COCO format')
parser.add_argument('--bckg_images', type=str, default=None, help='Root of background images', required=True)
parser.add_argument('--t_4_train_path', type=str, default=None, help='Root of Track4 data', required=True)
parser.add_argument('--store_path', type=str, default=None, help='Where to store data', required=True)
parser.add_argument('--classes_path', type=str, default=None, help='JSON file with classes', required=True)
parser.add_argument('--annotation_path', type=str, default=None, help='JSON file with generated annotations', required=True)
parser.add_argument('--count', type=int, default=100000, help='Count of generated data')

args = parser.parse_args()

bckg_images = os.listdir(args.bckg_images)

images = [os.listdir(os.path.join(args.t_4_train_path, 'syn_image_train', '{:05d}'.format(label))) for label in list(range(1,117))]
images.insert(0, [])

with open(args.classes_path, 'r') as f:
    categories = json.load(f)

annotations = {
        'info' : {
                    'description' : 'AIC2022 Track4 Dataset',
                    'url' : None,
                    'version' : '1.0',
                    'year' : 2022
                 },
        'licenses' : [None],
        'images' : [],
        'annotations' : [],
        'categories' : categories
        }



annot_id = 0

for pos in tqdm(range(args.count)):

    labels = np.random.randint(1, 117, np.random.randint(1,5))
    act_bckg = cv2.imread(os.path.join(args.bckg_images, random.choice(bckg_images)))
    h, w, _ = act_bckg.shape

    for l in labels:
        img_path = os.path.join(args.t_4_train_path, 'syn_image_train', '{:05d}'.format(l), random.choice(images[l]))
        seg_path = os.path.join(args.t_4_train_path, 'segmentation_labels', '{:05d}'.format(l), os.path.basename(img_path).split('.')[0]+'_seg.jpg')
        img = cv2.imread(img_path)
        seg = cv2.imread(seg_path)
        _, seg = cv2.threshold(seg, 128, 255, cv2.THRESH_BINARY)
        act_h, act_w, _ = img.shape

        x_pos = np.random.randint(1, w-act_w-1)
        y_pos = np.random.randint(1, h-act_h-1)

        crop = act_bckg[y_pos:y_pos+act_h, x_pos:x_pos+act_w, :]

        crop = np.where(seg, img, crop)
        act_bckg[y_pos:y_pos+act_h, x_pos:x_pos+act_w, :] = crop

        annotations['annotations'].append(
        {
            'image_id' : pos,
            'bbox' : [float(x_pos), float(y_pos), float(act_w), float(act_h)],
            'category_id' : int(l),
            'id' : annot_id,
            'iscrowd' : 0,
            'area' : float(act_w*act_h/2)
        })
        annot_id += 1

    annotations['images'].append(
    {
        'file_name' : '{:012d}.jpg'.format(pos),
        'width' : w,
        'height' : h,
        'id' : pos
    })
    cv2.imwrite(os.path.join(args.store_path, '{:012d}.jpg'.format(pos)), act_bckg)

with open(args.annotation_path, 'w') as f:
    json.dump(annotations, f)
