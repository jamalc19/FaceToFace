
# coding: utf-8

# In[32]:


import os
import numpy as np
from shutil import copy


# In[24]:


emote_dict = {0:"neutral",1:"anger",2:"contempt",3:"disgust",4:"fear",5:"happy",6:"sadness",7:"surprise"}
#0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise


# Copies over the peak emotions and neutral emotions to another folder.

# In[82]:


path = "Project\\Data\\Emotion_labels\\Emotion\\"
imgdirpath = "Project\\Data\\extended-cohn-kanade-images\\cohn-kanade-images\\"
newdir = "Project\\Parsed\\"

for p_id in os.listdir(path):
    p_hasNeutral = False
    
    for emote in os.listdir(path+p_id):
        
        if os.listdir(path+p_id+"\\"+emote) != None:
            
            #if folder for that person doesn't exist
            if os.path.isdir(newdir+p_id) == False:
                os.mkdir(newdir+p_id)
            
            #if emote_f for person p_id satisfied the FAC
            for emote_f in os.listdir(path+p_id+"\\"+emote):
                
                #read the type of emotion
                openf = open(path+p_id+"\\"+emote+"\\"+emote_f,"r")
                
                s = openf.read()
                s = int(s.split()[0][0:1])
                print(s)
                
                #get emotion from code
                emotion = emote_dict[s]
                print(path+p_id+"\\"+emote+"\\"+emote_f)
                print(emotion)
                
                #copy image over and rename with emotion suffix
                imgpath = imgdirpath+p_id+"\\"+emote+"\\"+os.listdir(imgdirpath+p_id+"\\"+emote)[-1]
                newpath = newdir+p_id+"\\"
                copy(imgpath,newpath)
                os.rename(newpath+os.listdir(imgdirpath+p_id+"\\"+emote)[-1],newpath+emotion+".png")
                
                #copy neutral over, if hasnt been copied yet
                if p_hasNeutral == False:
                    imgpath = imgdirpath+p_id+"\\"+emote+"\\"+os.listdir(imgdirpath+p_id+"\\"+emote)[0]
                    copy(imgpath,newpath)
                    os.rename(newpath+os.listdir(imgdirpath+p_id+"\\"+emote)[0],newpath+"neutral.png")
                    p_hasNeutral = True
 

