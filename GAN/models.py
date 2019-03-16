import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class_names = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'neutral']
classes_map = {'anger':0, 'disgust':1, 'fear':2, 'happy':3, 'sadness':4, 'surprise':5, 'neutral':6}
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()    
        
        self.encoder = nn.Sequential( 
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=7,stride=1,padding=3), #in channels is temporary
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
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=7,stride=1,padding=3),
            nn.Tanh()
        )
    
    def forward(self,x, target_emotions):
        #torch.eye(7)[target_emotions]
        
        #inject one hot encoding
        one_hot=torch.zeros(x.shape[0], 7,128,128)
        one_hot[range(x.shape[0]),target_emotions,:,:]=1
        x=torch.cat((x,one_hot),dim=1)
        
        
        x=self.encoder(x)
        #x=self.residual(x)????
        x=self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()    
        
        self.initiallayers = nn.Sequential( 
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(negative_slope=0.2))
            
        self.finallayers = nn.Sequential( 
            nn.Conv2d(in_channels=71, out_channels=128, kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128,momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2),           
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256,momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(512,momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2),   
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=10,stride=1,padding=1), #a fc layer
            F.sigmoid()
            )        
        
    def forward(self,x, target_emotions):
        x=self.initiallayers(x)
        
        #inject one hot encoding
        one_hot=torch.zeros(x.shape[0], 7,64,64)
        one_hot[range(x.shape[0]),target_emotions,:,:]=1  
        x=torch.cat((x,one_hot),dim=1)        
        
        x=self.finallayers(x)
        return x.reshape(x.shape[0],1) #batchsize, output
    
    

   
    
    #############################################################
#Classifier code from https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch
    
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
        out=F.interpolate(x, size=48,mode='bilinear', align_corners=False)#Jamal Resize
        out = self.features(out)
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
    
class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()    
        self.alex  = torchvision.models.alexnet(pretrained=True).features
        
    def forward(self,x):
        return self.alex(x)
        


generator=Generator()
discriminator = Discriminator()

def get_data(batch_size=32):
    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
    neutralset = torchvision.datasets.ImageFolder('../train/neutral', transform=transform)
    
    neutral_loader = torch.utils.data.DataLoader(neutralset, batch_size=batch_size,
                                              num_workers=1, shuffle=True)


        
    emotionset = torchvision.datasets.ImageFolder('../train/emotions', transform=transform)
    dataloaderclasses = emotionset.classes
    dataloaderclasses_map=emotionset.class_to_idx
    
    emotion_loader=torch.utils.data.DataLoader(emotionset, batch_size=batch_size,
                                              num_workers=1, shuffle=True)
    
    return neutral_loader, emotion_loader, dataloaderclasses, dataloaderclasses_map



    

#load saved parameters
def train(generator,discriminator, batchsize=32,lr=0.001, gan_loss_weight=75, identity_loss_weight=0.5e-4, emotion_loss_weight=30): 
    torch.manual_seed(1000)#set random seet for replicability
    
    #set to train mode for batch norm calculations
    generator.train(mode=True)
    discriminator.train(mode=True)
    
    classifier = Classifier()
    featurenet = FeatureNet()    
    classifier.eval()
    feature.eval()
    
    optimizerG=optim.Adam(generator.parameters(), lr=lr)#add hyperparameters
    optimizerD=optim.Adam(discriminator.parameters(), lr=lr)
    
    ClassifierCriterion = nn.CrossEntropyLoss()
    DiscriminatorCriterion = nn.MSELoss()
    FeatureCriterion = nn.MSELoss()
    
    neutral_loader, emotion_loader, dataloaderclasses, dataloaderclasses_map = get_data(batch_size=32)
    
    for epoch in range(num_epochs):
        for emotion_pics, labels in emotion_loader:
            
            
            
            neutral_pics = next(iter(neutral_loader))[0]
            
            labels = torch.tensor([classes_map[dataloaderclasses[i]] for i in labels]) #convert labels from dataloader indices to model indices
            
            fakelabels = ((labels+np.random.randint(-10,10))*np.random.randint(1,7)) %7 #roughly a uniform transformation from any i in range(0,6) to j in range(0,6)
            
            #forward pass
            generated_pics = generator(neutral_pics, labels)
            generated_out = discriminator(generated_pics, labels)
            real_out = discriminator(emotion_pics, labels)
            fakelabel_out = discriminator(emotion_pics, fakelabels)
            
            #feature loss
            neutral_features = featurenet(neutral_pics)
            generated_features = featurenet(generated_features)
            feat_loss = FeatureCriterion(generated_features,neutral_features)
            
            #classifier loss
            class_preds= classifier(generated_features)
            classifier_loss = ClassifierCriterion(class_preds, labels)
            
            #GAN loss
            D_loss = (0.5*torch.mean((real_out - 1)**2) + 0.25*torch.mean(generated_out**2) + 0.25*torch.mean(fakelabel_out**2)) *gan_loss_weight
            G_loss = 0.5*torch.mean((generated_out-1)**2)
            
            totalG_loss = gan_loss_weight*G_loss + identity_loss_weight*feat_loss + emotion_loss_weight*classifier_loss
        
            
            #backward pass
            optimizerD.zero_grad()
            D_loss.backward()
            optimizerD.step()
            
            optimizerG.zero_grad()
            totalG_loss.backward()
            optimizerG.step()
        
    
    
    
    
    
    
    #training done put in eval mode
    discriminator.eval()
    generator.eval()
    
    

        
