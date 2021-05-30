import torch
import torchvision
from torchvision import transforms, models
from pathlib import Path
import io
from PIL import Image

class data_preprocessing_for_net():   
    class_names = ['BAC_PNEUMONIA', 'NORMAL','VIR_PNEUMONIA']
    
    def train_valid_preprocessing(self, train_dir = "Pneumonia_data/data/train", val_dir = "Pneumonia_data/data/validation"):
        batch_size = 23 #or 12
        data_to_tranform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) 
        train_dataset = torchvision.datasets.ImageFolder(train_dir, data_to_tranform)
        val_dataset = torchvision.datasets.ImageFolder(val_dir, data_to_tranform)
        #droplast - drop last batch if it's not 23
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
        val_dataloader = torch.utils.data.DataLoader(
           val_dataset, batch_size=batch_size, shuffle=False, drop_last = True)
        return train_dataloader, val_dataloader

    def transform_image_to_predict(self, image):
        my_transforms = transforms.Compose([transforms.Resize(255),#mb out this function
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                [0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])
        
       # image = Image.open(image)#comment it!!!!!!!!!!!!!!!!!!!!!!!!!
        image = image.convert("RGB")
        return my_transforms(image).unsqueeze(0)
    
    def find_accuracy(self, test_dir = "Pneumonia_data/data/test"):
        batch_size = 23
        data_to_tranform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) 
        test_dataset = ImageFolderWithPaths(test_dir, data_to_tranform)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
        return test_dataloader

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path