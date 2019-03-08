
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np


# In[65]:


data = datasets.ImageFolder("C:/Users/Ling/Documents/GitHub/FaceToFace/Cropped",transform=transforms.ToTensor())


# In[24]:


c = transforms.ToPILImage()


# In[70]:


samples = [data[0][0],data[2][0]]


# In[20]:


sample = data[0][0]


# In[35]:


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
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


# In[62]:


y = model(sample)
c(y[0])


# In[77]:


def overfit(model,data, num_epochs=5, batch_size=16, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    outs = []
    
    loader = torch.utils.data.DataLoader(data, batch_size=1,
                                                  num_workers=1, shuffle=True)   
    
    for epoch in range(num_epochs):
        for sample in loader:
      
            recon = model(sample)
            loss = criterion(recon, sample)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch%5 == 0:
            outs.append(recon)
            print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))


    return 


# In[90]:


#2 sample model
model2=Autoencoder()
overfit(model2,samples,50)


# In[89]:


c(model2(samples[1].unsqueeze(0))[0])


# In[87]:


c(model2(samples[0].unsqueeze(0))[0])


# ## Testing unseen faces

# In[98]:


c(model2(data[90][0].unsqueeze(0))[0])


# In[57]:


#1 sample model
model = Autoencoder()
overfit(model,sample,150)

