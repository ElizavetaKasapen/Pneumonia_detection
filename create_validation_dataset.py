import shutil #operations with files and dir 
from tqdm import tqdm #progress bar
import os
import zipfile

#unpacking dataset
zip_path = 'Pneumonia_data.zip'
folder_path = 'Pneumonia_data'
with zipfile.ZipFile(zip_path, 'r') as zip_obj:
   zip_obj.extractall(folder_path)
   
#main paths
train_dir = 'Pneumonia_data/data/train'
val_dir = 'Pneumonia_data/data/validation'
class_names = ['BAC_PNEUMONIA', 'NORMAL','VIR_PNEUMONIA']

#every 5 pic to the valid set
def creating_valid_dataset(each_img = 5):
  for class_name in class_names:
     os.makedirs(os.path.join(dir, class_name), exist_ok=True)
     source_dir = os.path.join(train_dir, class_name)
     for i, file_name in enumerate(tqdm(os.listdir(source_dir))):
        if i % each_img == 0:
            dest_dir = os.path.join(dir, class_name)
            shutil.move(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))