import torch
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as func
import data_processing

class nn_for_pneumonia_detection():

    

    def create_model(self, number_of_classes = 3):#4 with unknown, but for v_01 - 3
        self.model = models.densenet121(pretrained=True)
        # Disable grad for all conv layers
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, number_of_classes)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        return self.model

    def to_train(self, path = "state_dict_model.pt"):#path to save trained model
        self.writer = SummaryWriter()
        self.dataloaders = data_processing.data_preprocessing_for_net()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = self.create_model()
        #print(net)
        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3)
        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        net = self.train_model(self.dataloaders, net, loss, optimizer, scheduler, num_epochs=1)
        self.writer.flush()
        self.writer.close()
        net.eval()
        torch.save(net.state_dict(), path)

    def train_model(self,  dataloaders, model, loss, optimizer, scheduler, num_epochs):
        train_dataloader, val_dataloader = dataloaders.train_valid_preprocessing()
        print(len(train_dataloader))
        x_batch, y_batch = next(iter(train_dataloader))
        #if you want to show model
        self.writer.add_graph(model, x_batch)

        #epoch_loss_dict =  dict.fromkeys(['train', 'val'], [])
    
        for epoch in range(num_epochs):
            print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    dataloader = train_dataloader
                    #scheduler.step() # documentation said I have to do it after optim
                    model.train()  # Set model to training mode
                else:
                    dataloader = val_dataloader
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.
                running_acc = 0.

                # Iterate over data.
                for inputs, labels in tqdm(dataloader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    # forward and backward
                    with torch.set_grad_enabled(phase == 'train'):
                        preds = model(inputs)
                        loss_value = loss(preds, labels)
                        preds_class = preds.argmax(dim=1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss_value.backward()
                            optimizer.step()
                            scheduler.step()
                    # statistics
                    running_loss += loss_value.item()
                    running_acc += (preds_class == labels.data).float().mean()

                epoch_loss = running_loss / len(dataloader)
                epoch_acc = running_acc / len(dataloader)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)
                #for epoch loss
                self.writer.add_scalar("Loss/{}".format(phase), epoch_loss, epoch)
                #for epoch acc
                self.writer.add_scalar("Acc/{}".format(phase), epoch_acc, epoch)
        #to see overfitting
                #epoch_loss_dict[phase].append(epoch_loss) # mb. CHECK IT!
                #if phase == 'train':
                   # epoch_loss_arr_train.append(epoch_loss)
                #else:
                    #epoch_loss_arr_val.append(epoch_loss)
            #self.writer.add_scalars('Loss',{'train':epoch_loss_dict['train'],'validation':epoch_loss_dict['val']},epoch)
        return model
    
    def load_model(self, path = "state_dict_model.pt"):
        model = self.create_model()
        model.load_state_dict(torch.load(path))
        model.eval() 
        #print(model.classifier)
        return model

#add path
    def to_predict(self, path):
        data_transform = data_processing.data_preprocessing_for_net()
        tensor = data_transform.transform_image_to_predict(path)
        print(tensor)
        model = self.load_model()
        #or maybe outputs = model.forward(tensor)
        with torch.set_grad_enabled(False):
            outputs = model(tensor)
            outputs = func.softmax(outputs, dim=1)
        prob = list(outputs) #change
        return y_hat
        
