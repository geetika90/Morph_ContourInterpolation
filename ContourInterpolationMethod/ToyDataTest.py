#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
from skimage.morphology import (square, disk,
                                 ball,  )
import sys
sys.path.insert(1,'/ContourInterpolationMethod')
from ContourInterpolationMethod.All_modules_redifened  import *






from skimage.morphology import (square, disk,
                                 ball,  octagon)

from ContourInterpolationMethod.TestValidation import IntContourVal 
from skimage.io import imread,imsave

import matplotlib





get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:



def InterpToyData(fname):
    Image = imread(fname+'.tif',as_gray='false')
    

    se = disk(1)
    iteration = 3
    perc =60
    Bseed,hseed,helv= getSeed(fname)
    workingimg ,Elv,NoData = PreproWithoutHill(Image,Bseed,hseed,helv,se)
    BinaryInput = ThrDecomp(Image,workingimg,perc)
    workingimg_mod,Mod_InpImg,BinaryInput = Get_hill_set(Image,workingimg,BinaryInput,hseed,helv,50,se,NoData)
    cont_iter = IntContourVal(Image,workingimg_mod,BinaryInput,se,iteration=3)
    import matplotlib
    import matplotlib_inline
    dpi = matplotlib.rcParams['figure.dpi']
    # figsize  =  1* Image.shape[1]/float(dpi),1*Image.shape[0]/float(dpi)

    row,col = 2, max(len(BinaryInput),len(cont_iter))+1
    fig, axes = plt.subplots(row,col,figsize=(20,8))

    fig.suptitle('Input Toy data and its interpolated intermediate contours ',fontsize=15)

    for ax in axes.flatten():
        ax.axis('off')
    axes[0][0].imshow(Image)
    axes[0][1].imshow(BinaryInput[0][0])
    axes[0][2].imshow(BinaryInput[1][0])
    axes[0][3].imshow(BinaryInput[1][1])

    for i in range(len(cont_iter)):
        axes[1][i].imshow(cont_iter[i])
        axes[1][i].title.set_text( 'LEVEL' + str(i+1))


    plt.show()        


# In[ ]:




