# PersonGONE
Implementaion of PersonGONE

# TODOs:
Zkus, zda cele funguje   
Nastavit spravne velikosti datasetu a epochy tranovani (ted jen na zkousku)    
Pridat parametry do detect_and_create_submission - tsize, tracker, expn    

# Installation
Install CUDA 11.3 \
virtualenv _env \
source _env/bin/activate \
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html \
pip install opencv-python scikit-image tqdm Pillow thop ninja loguru tabulate pycocotools tensorboard filterpy lap \
pip install mmcv-full==1.4.6 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html \
pip install mmdet \
pip install pyyaml numpy easydict==1.9.0 scikit-learn joblib matplotlib pandas albumentations==0.5.2 hydra-core kornia==0.5.0 webdataset packaging wldhx.yadisk-direct \
pip install cython cython_bbox pytorch-lightning wget

# Download pretrained models
Popis

# Training
Popis

# Inference
Popis
