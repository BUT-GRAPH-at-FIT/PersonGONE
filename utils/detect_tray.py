import cv2
import numpy as np
import json
import argparse


parser = argparse.ArgumentParser('ROI detection process!')
parser.add_argument('--mean_path', type=str, default=None, help='Path to mean image')
parser.add_argument('--out_path', type=str, default=None, help='Path to output JSON file')
parser.add_argument('--roi_seed', type=int, nargs=2, default=None, help='Seed to found ROI')
args = parser.parse_args()

path = args.mean_path
store_path = args.out_path
img = cv2.imread(path)
h, w, _ = img.shape

grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

scale = 1
delta = 0
ddepth = cv2.CV_16S
grad_x = cv2.Scharr(grayscale, ddepth, 1, 0, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Scharr(grayscale, ddepth, 0, 1, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


seed = (int(w/2), int(h/2)) if args.roi_seed is None else (int(args.roi_seed[0]), int(args.roi_seed[1]))

ret, out_img, mask, rect = cv2.floodFill(grad, np.zeros((h+2,w+2), dtype=np.uint8), seed, 255, 7, 7)

out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
out_img = cv2.rectangle(out_img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,0,255), 7)
with open(store_path, 'w') as f:
    json.dump([rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]], f)
