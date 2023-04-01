#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np


# In[5]:


def PreproWithoutHill(Image,Bseed,hseed,helv,se):
    inptimage1 =Image.copy()
#     inptimage2 =Image.copy()
    
    
    Elv = np.unique(Image)  #------Calculate no of regions from the original Image-------#
    NoData = Elv[0]
    Elv = np.delete(Elv,0)
    No_Reg = len(Elv)
    Hseed = [(hseed[i],helv[i]+1) for i in range(0,len(hseed))]
    Bseed.extend(Hseed)
    workingimg = preprocess_fill(inptimage1, Bseed, c=4,elevation=1)
#     workingimg,inptimage,Helv,HillSeed  = HillTopPreprocess(inptimage2, workingimg, hseed,helv,se,20)
    
    
    
    
    return   workingimg,Elv,NoData
    


# In[ ]:





# In[ ]:




