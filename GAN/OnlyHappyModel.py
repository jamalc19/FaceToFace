import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle


class_names = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'neutral']
classes_map = {'anger':0, 'disgust':1, 'fear':2, 'happy':3, 'sadness':4, 'surprise':5, 'neutral':6}
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()    
        
        self.encoder = nn.Sequential( 
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7,stride=1,padding=3), #in channels is temporary
            nn.BatchNorm2d(32,momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64,momentum=0.1),
            nn.ReLU(),            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128,momentum=0.1),
            nn.ReLU()
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size=3,padding=1, stride=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(),       
            nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size=3,padding=1, stride=1),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=7,stride=1,padding=3), #switch to 1 channel for black and white
            nn.Tanh()
        )
        
        self.residual = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size=3,padding=1, stride=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(),       
            
            nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size=3,padding=1, stride=1),
            nn.BatchNorm2d(128, momentum=0.1)           
            )

    
    def forward(self,x, target_emotions, residual_blocks=6, cuda=True):
        x=self.encoder(x)
            
        for i in range(residual_blocks):
            x = x + self.residual(x)
            
        x=self.decoder(x)#output is black and white
        x=torch.cat((x,x,x),dim=1) #convert to 3 channels
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()    
        
        self.initiallayers = nn.Sequential( 
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(negative_slope=0.2))
            
        self.finallayers = nn.Sequential( 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128,momentum=0.1),
            nn.LeakyReLU(negative_slope=0.2),           
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256,momentum=0.1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(512,momentum=0.1),
            nn.LeakyReLU(negative_slope=0.2),   
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=10,stride=1,padding=1) #a fc layer
            )        
        
    def forward(self,x):
        x=self.initiallayers(x)
        x=self.finallayers(x)
        x = x.reshape(x.shape[0],1) #batchsize, output
        return torch.sigmoid(x)

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()    
        self.alex  = torchvision.models.alexnet(pretrained=True).features
        
    def forward(self,x):
        return self.alex(x)

def get_data(batch_size=32, overfit=False, colab=True):
    '''
    gets dataloaders for neutral and emotion datasets.
    '''
    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    #Normalize outside
    
    
    path='../train/neutral'
    if colab:
        path="FaceToFace/train/neutral"
    if overfit:
        path+='overfit'
    
    neutralset = torchvision.datasets.ImageFolder(path, transform=transform)
    
    neutral_loader = torch.utils.data.DataLoader(neutralset, batch_size=batch_size,
                                              num_workers=1, shuffle=True, drop_last=True)

    path='../train/HappyFolder'
    if colab:
        path="FaceToFace/train/HappyFolder"
    if overfit:
        path+='overfit'

    emotionset = torchvision.datasets.ImageFolder(path, transform=transform)
    dataloaderclasses = emotionset.classes
    
    emotion_loader=torch.utils.data.DataLoader(emotionset, batch_size=batch_size,
                                              num_workers=1, shuffle=True, drop_last=True)
    
    return neutral_loader, emotion_loader, dataloaderclasses

#load saved parameters
def train(generator,discriminator, num_epochs,checkpointfolder='', batchsize=32,lr=0.001,
          gan_loss_weight=75, identity_loss_weight=0.5e-3, overfit=False, colab=True, start_epoch=0):
    torch.manual_seed(1000)#set random seet for replicability

    #set to train mode for batch norm calculations
    generator.train(mode=True)
    discriminator.train(mode=True)
    
    Losses={}
    Losses['G_Losses']=[]
    Losses['Feature_Losses']=[]
    
    Losses['D_Real_Losses']=[]
    Losses['D_Fake_Losses']=[]
    Losses['D_Generator_Losses']=[]
    
    Losses['TotalD_Losses']=[]
    Losses['TotalG_Losses']=[]

    # featurenet always in eval mode
    featurenet = FeatureNet()
    featurenet.eval()
    
    optimizerG=optim.Adam(generator.parameters(), lr=lr) #add hyperparameters
    optimizerD=optim.Adam(discriminator.parameters(), lr=lr)
    
    if colab:
       
        generator.cuda()
        discriminator.cuda()
        featurenet.cuda()
    
    DiscriminatorCriterion = nn.BCELoss()
    FeatureCriterion = nn.MSELoss()
    fake_label = 0
    real_label = 1
    
    neutral_loader, emotion_loader, dataloaderclasses = get_data(batch_size=batchsize, overfit=overfit, colab=colab)
    
    for epoch in range(start_epoch+1,start_epoch+num_epochs+1):
        epoch_g_loss=0
        epoch_d_loss=0
        for emotion_pics, labels in emotion_loader:
            
            neutral_pics = next(iter(neutral_loader))[0]
            
            labels = torch.tensor([classes_map[dataloaderclasses[i]] for i in labels]) #convert labels from dataloader indices to model indices
            
            if colab:
                emotion_pics = emotion_pics.cuda()
                neutral_pics = neutral_pics.cuda()
                labels = labels.cuda()

            #forward pass for generator
            optimizerG.zero_grad()
            ######################################################################
            generated_pics = generator(neutral_pics, labels, cuda=colab)
            generated_out= discriminator(generated_pics)
            
            #feature loss
            neutral_features = featurenet(neutral_pics) 
            generated_features = featurenet(generated_pics)
            feat_loss = FeatureCriterion(generated_features,neutral_features)*identity_loss_weight #both input images are normalized

            #GAN loss      
            #G_loss = torch.mean((generated_out-1)**2)
            
            G_loss = DiscriminatorCriterion(generated_out, torch.ones_like(generated_out))*gan_loss_weight
            
            totalG_loss = G_loss + feat_loss
            totalG_loss.backward()
            optimizerG.step()
            
            Losses['G_Losses'].append(G_loss)
            Losses['Feature_Losses'].append(feat_loss)
            Losses['TotalG_Losses'].append(totalG_loss)
            
            ###################################################################
            #Discriminator Pass
            optimizerD.zero_grad()

            generated_out= discriminator(generated_pics.clone().detach()) #discriminator takes normalized images -1,1.
            real_out = discriminator(emotion_pics)#emotion loader already normalizes pics

            
            generatedD_loss = 0.5*DiscriminatorCriterion(generated_out,torch.zeros_like(generated_out)) *gan_loss_weight
            realD_loss =  0.5*DiscriminatorCriterion(real_out, torch.ones_like(real_out)) *gan_loss_weight

            D_loss=generatedD_loss + realD_loss
                 
            D_loss.backward()
            optimizerD.step()
            
            Losses['D_Real_Losses'].append(realD_loss)
            Losses['D_Generator_Losses'].append(generatedD_loss)
            Losses['TotalD_Losses'].append(D_loss)

            #print("G",float(totalG_loss))
        if epoch%25 == 0 or colab:
            d_params= discriminator.state_dict()
            g_params = generator.state_dict()
                

            path='checkpoints'
            if colab:
                path ="/content/gdrive/My Drive/APS360/Checkpoints"+checkpointfolder
                if epoch%5==0:
                    print("Epoch", epoch)
                    pickle.dump(Losses,open(path+'/Losses'+str(epoch), 'wb'))
                    torch.save(d_params, path+'/discriminator'+str(epoch))
                    torch.save(g_params, path+'/generator'+str(epoch))
            
    #training done put in eval mode  
    discriminator.eval()
    generator.eval()
    
    plot_loss(Losses)
    
    return Losses

    
def plot_loss(Losses):
    plt.title("Losses")
    losses=list(Losses.keys())
    
    size = len(Losses[losses[0]])
    for loss in losses:
        plt.plot(range(size), Losses[loss], label=loss)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='best')
    plt.show()


def visualize_sample(generator,  colab=True):
    neutral_loader, emotion_loader, dataloaderclasses = get_data(batch_size=1, overfit=False, colab=colab)
    if colab:
        generator.cuda()
    for neutral_pics, labels in neutral_loader:
        if colab:
            neutral_pics=neutral_pics.cuda()
            labels=labels.cuda()

        neutral=(neutral_pics +1)/2
        
    
        fig, axes = plt.subplots(2,4)
        axes[0][0].imshow(np.transpose(neutral[0,:,:,:].detach(), [1,2,0]), cmap='gray')
        axes[0][0].set_title('Neutral (input)')        
        for i in range(6):
            emotion = class_names[i]
            labels[:] = i
            out = generator(neutral_pics, labels,cuda=colab)
            out = out*0.5+0.5
            
            ax=axes[(i+1)//4][(i+1)%4]
            ax.imshow(np.transpose(out[0,:,:,:].detach(), [1,2,0]), cmap='gray')
            ax.set_title(emotion)    
        plt.show()
        break

    ''' 
    im1=np.transpose(out[0,:,:].detach(), [1,2,0])
    im2=np.transpose(out[1,:,:].detach(), [1,2,0])
    plt.imshow(im1, cmap='gray')
    plt.show()
    plt.imshow(im2, cmap='gray')
    plt.show()  
    

    print(class_names[labels[0]])
    print(class_names[labels[1]])
    out=generator(neutral_pics, labels,cuda=colab)
    im1=np.transpose(out[0,:,:].detach(), [1,2,0])
    im2=np.transpose(out[1,:,:].detach(), [1,2,0])
    plt.imshow(im1, cmap='gray')
    plt.show()
    plt.imshow(im2, cmap='gray')
    plt.show()
    
    out=(emotion_pics +1)/2
    im1=np.transpose(out[0,:,:].detach(), [1,2,0])
    im2=np.transpose(out[1,:,:].detach(), [1,2,0])
    plt.imshow(im1, cmap='gray')
    plt.show()
    plt.imshow(im2, cmap='gray')
    plt.show()
    '''

def load_loss(epoch, subfolder):
    return pickle.load(open("/content/gdrive/My Drive/APS360/Checkpoints"+subfolder+"/Losses"+str(epoch), 'rb'))

def load_model(epoch, subfolder,colab=True):
    '''
    subfolder in  "/Jamal", "/Ling", "/Michael"
    '''
    generator=Generator()
    if colab:
        path ="/content/gdrive/My Drive/APS360/Checkpoints"+subfolder+"/generator" 
        weights = torch.load(path+str(epoch))
    else:
        path='checkpoints/generator'
        weights = torch.load(path+str(epoch), map_location='cpu')
    generator.load_state_dict(weights)    
    
    discriminator = Discriminator()    
    if colab:
        path ="/content/gdrive/My Drive/APS360/Checkpoints"+subfolder+"/discriminator"    
        weights = torch.load(path+str(epoch))
    else:
        path='checkpoints/discriminator'
        weights = torch.load(path+str(epoch), map_location='cpu')
    discriminator.load_state_dict(weights)    
    return generator, discriminator

    
if __name__=="__main__":
    generator=Generator()
    discriminator = Discriminator()    
    Losses = train(generator,discriminator,checkpointfolder='/Michael', num_epochs=4, batchsize=2,lr=0.001, gan_loss_weight=30,
                   identity_loss_weight=0.5e-3, overfit=False, colab=False, start_epoch=20)
