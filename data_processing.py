import torch
import torchvision
from torchvision import transforms, models
from pathlib import Path

class data_preprocessing_for_net():   
    class_names = ['BAC_PNEUMONIA', 'NORMAL','VIR_PNEUMONIA']
    
    def train_valid_preprocessing(self, train_dir = "/Pneumonia_data/data/train", val_dir = "/Pneumonia_data/data/validation"):
        batch_size = 23 #or 12
        data_to_tranform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        #mb change later
        train_dir = Path(Path.cwd(), "Pneumonia_data","data","train")
        val_dir = Path(Path.cwd(), "Pneumonia_data","data","validation")
        train_dataset = torchvision.datasets.ImageFolder(train_dir, data_to_tranform)
        val_dataset = torchvision.datasets.ImageFolder(val_dir, data_to_tranform)
    #droplast - drop last batch if it's not 23
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
        val_dataloader = torch.utils.data.DataLoader(
           val_dataset, batch_size=batch_size, shuffle=False, drop_last = True)
        return train_dataloader, val_dataloader