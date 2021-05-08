import torch
import torchvision.models as models

#class Net(torch.nn.Module):
    #def __init__(self, number_of_classes = 3):#4 with unknown, but for v_01 - 3
        #super(Net, self).__init__()
class Net():
    def create_model(self, number_of_classes = 3):#4 with unknown, but for v_01 - 3
        self.model = models.densenet121(pretrained=True)
        # Disable grad for all conv layers
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, number_of_classes)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        return self.model
    
