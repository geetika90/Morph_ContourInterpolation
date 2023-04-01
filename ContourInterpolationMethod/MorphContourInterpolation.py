#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from skimage.morphology import (square, disk,
                                 ball,  )
import sys
sys.path.insert(1,'/ContourInterpolationMethod')
from ContourInterpolationMethod.All_modules_redifened  import *



# from ContourInterpolationMethod.TestValidation import * 


# In[ ]:


def getInterpolSurf(InputImage,ImgName):
    
    workingimg_org_mod,InputImage_mod,BinaryInput_org,se = getThresholdDecompos(InputImage,ImgName)
    cont_org = IntContour1(InputImage_mod, workingimg_org_mod, BinaryInput_org,se)
    test_org = cont_org.copy()
    result_org = postprocessing1(test_org)
#     plt.imsave('../'+ ImgName+' Result.tiff',result_org,cmap='plasma')
#     joblib.dump( result_org,'../'+ ImgName + ' Result.z')
    return result_org


# In[ ]:


def getThresholdDecompos(InputImage,ImgName):
    se =  disk(1)
    perc = 60
    
    Bseed, hseed,helv = getSeed(ImgName)
    workingimg_org,Elv_org ,NoData = PreproWithoutHill(InputImage,Bseed,hseed,helv,se)
    BinaryInput_org = ThrDecomp(InputImage,workingimg_org,perc)
    workingimg_org_mod ,InputImage_mod ,BinaryInput_org = Get_hill_set(InputImage,workingimg_org,BinaryInput_org,hseed,helv,20,se,NoData)
    
    return workingimg_org_mod,InputImage_mod,BinaryInput_org,se
    
    


# In[ ]:





# In[ ]:




