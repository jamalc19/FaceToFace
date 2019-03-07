import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(model, num_epochs=5, batch_size=16, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    data = datasets.ImageFolder('Parsed', transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                  num_workers=1, shuffle=True)    
   

    for epoch in range(num_epochs):
        for data in loader:
            img, label = data
            img =img[:,0:1,:,:]
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))

    return 

def displayresults(model, num=10):
    data = datasets.ImageFolder('Parsed', transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(data, batch_size=1,
                                                  num_workers=1, shuffle=True)        
    i=0
    for sample in loader:
        img, label = sample
        #img=img[:,0,:,:]#images are black and white but in RGB. All channels are identical
        img =img[:,0:1,:,:]
        recon = model(img).detach().numpy()
        if i >= num:
            break
        
        img=img[0,0,:,:]
        
        recon=recon[0,0,:,:]
        plt.subplot(2, num, i+1)
        plt.imshow(img)       
        plt.subplot(2, num, num+i+1)
        plt.imshow(recon)
        i+=1
    plt.show()



if __name__ == "__main__":
    model = Autoencoder()
    max_epochs = 5
    train(model, num_epochs=max_epochs)
    #displayresults(model, num=10)
