import argparse
import os
from subprocess import call


parser = argparse.ArgumentParser('Create dataset for YOLOX training')
parser.add_argument("--t_1_path", type=str, default=None, help="Path to root of Track1 data", required=True)
parser.add_argument("--t_3_path", type=str, default=None, help="Path to root of Track3 data", required=True)
parser.add_argument("--t_4_path", type=str, default=None, help="Path to root of Track4 data", required=True)

args = parser.parse_args()

data_path = './data'
abs_data_path = os.path.abspath(data_path)
dataset_path = os.path.join(abs_data_path, 'dataset')

os.makedirs(dataset_path, exist_ok=True)

print('Extracting background images from Track1 and Track3!')
call(['python', 'utils/extract_bckg_images.py',
      '--t_1_path', args.t_1_path,
      '--t_3_path', args.t_3_path,
      '--data_path', abs_data_path])
print('Background images extraction DONE! Images stored in:', os.path.join(abs_data_path, 'bckg_images'))


print('Creating training dataset in COCO format')
os.makedirs(os.path.join(dataset_path, 'annotations'), exist_ok=True)
os.makedirs(os.path.join(dataset_path, 'train'), exist_ok=True)
call(['python', 'utils/generate_dataset_coco_format.py',
      '--bckg_images', os.path.join(abs_data_path, 'bckg_images'),
      '--t_4_train_path', os.path.join(args.t_4_path, 'Train_SynData'),
      '--store_path', os.path.join(dataset_path, 'train'),
      '--classes_path', os.path.join(abs_data_path, 'classes.json'),
      '--annotation_path', os.path.join(dataset_path, 'annotations', 'train.json'),
      '--count', '100'])
print('Train dataset DONE! Stored in:', os.path.join(dataset_path, 'train'))


print('Creating validation dataset in COCO format')
os.makedirs(os.path.join(dataset_path, 'validation'), exist_ok=True)
call(['python', 'utils/generate_dataset_coco_format.py',
      '--bckg_images', os.path.join(abs_data_path, 'bckg_images'),
      '--t_4_train_path', os.path.join(args.t_4_path, 'Train_SynData'),
      '--store_path', os.path.join(dataset_path, 'validation'),
      '--classes_path', os.path.join(abs_data_path, 'classes.json'),
      '--annotation_path', os.path.join(dataset_path, 'annotations', 'validation.json'),
      '--count', '20'])
print('Validation dataset DONE! Stored in:', os.path.join(dataset_path, 'validation'))
