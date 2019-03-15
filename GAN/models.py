import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



    

    
class Generator(nn.Module):
    def __init(self):
        super(Generator, self).__init__()    
        
        self.encoder = nn.Sequential( 
            nn.Conv2d(in_channels=9, out_channels=32, kernel_size=7,stride=1,padding=3), #in channels is temporary
            nn.BatchNorm2d(32,momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64,momentum=0.9),
            nn.ReLU(),            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128,momentum=0.9),
            nn.ReLU()
            )
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size=3,padding=1, stride=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(),       
            nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size=3,padding=1, stride=1),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=7,stride=1,padding=1),
            nn.Tanh()
        )
    
    def forward(self,x):
        x=self.encoder(x)
        #x=self.residual(x)????
        x=self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init(self):
        super(Discriminator, self).__init__()    
        
        self.layers = nn.Sequential( 
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128,momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2),           
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256,momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,stride=1,padding=1),
            nn.BatchNorm2d(512,momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2),   
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4,stride=1,padding=1),
            )        
        
    def forward(self,x):
        return self.layers(x)
    #output shape should be 1 or 2??
    
    #############################################################
#Classifier code from https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch
    
#from torch.autograd import Variable


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class Classifier(nn.Module):#accepts images 48/48 for now
    def __init__(self, vgg_name='VGG19'):
        super(Classifier, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 7)
        
        weights = torch.load('ClassifierWeights', map_location='cpu')['net']
        self.load_state_dict(weights)
        

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)    
    
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    ########################################
    
class FeatureNet(nn.Module):
    def __init(self):
        super(FeatureNet, self).__init__()    
        self.alex  = torchvision.models.alexnet(pretrained=True)
    
    


class GAN(nn.Module):
    
    def __init(self):
        super(GAN, self).__init__()
        
        self.generaotr=Generator()
        self.discriminator = Discriminator()
        self.classifier = Classifier()
        self.feature = FeatureNet()
        
        
        
#test classifier
'''
from torchvision import datasets


t = transforms.Compose ( [transforms.Resize(48),transforms.ToTensor()])
data = datasets.ImageFolder("../Cropped",transform=t)

loader = torch.utils.data.DataLoader(data, batch_size=1,
                                              num_workers=1, shuffle=True)  


f = open("../SupplementalData/fer2013.csv",'r')
import csv
reader = csv.reader(f)
next(reader)

im=next(reader)

    
x = im[1].split(' ')
visualize_x = np.array(x).astype(int).reshape(48,48)
x=np.array([x]*3).astype(np.float32)
x=torch.from_numpy(x.reshape(1,3,48,48))


plt.imshow(visualize_x, cmap='gray')
plt.show()
'''
