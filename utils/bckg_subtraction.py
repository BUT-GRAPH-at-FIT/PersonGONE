import cv2
import numpy as np
from tqdm import tqdm
import sys
import os

# path = '/mnt/matylda0/ispanhel/Datasets/AIC2022/AIC22_Track4_AutoRetail/TestA/inpainting_9_3/testA_5/testA_5_inpainting.mp4'
path = sys.argv[1]
store_path = sys.argv[2]

cap = cv2.VideoCapture(path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

bckg_sub = cv2.createBackgroundSubtractorMOG2()

images = []

for _ in tqdm(range(frames_cnt+1)):
    ret, frame = cap.read()
    if not ret:
        break

    fg_mask = bckg_sub.apply(frame)
    bg_image = bckg_sub.getBackgroundImage()

    images.append(bg_image)

imgs_array = np.array(images)

mean_img = np.mean(np.array(images), axis=0).astype(np.uint8)

cv2.imwrite(store_path, mean_img)

cap.release()
