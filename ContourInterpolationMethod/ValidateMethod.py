#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
from skimage.morphology import (square, disk,
                                 ball,  )
import sys
sys.path.insert(1,'/ContourInterpolationMethod')
from ContourInterpolationMethod.All_modules_redifened  import *
from ContourInterpolationMethod.MorphContourInterpolation  import getInterpolSurf
from ContourInterpolationMethod.TestValidation  import *

# from  All_modules_redifened import *
# from TestValidation import *

from hausdorff import hausdorff_distance
from skimage import metrics
import matplotlib
import pandas as pd
import plotly.io as pio

pio.renderers.default = 'notebook'
# import matplotlib.pyplot as plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


def Get_error(image, result, newelv ):
    import math
    from sklearn.metrics import mean_absolute_percentage_error
    
    RMSE =[]
    MAPE = []
    
    
    for i in newelv:
        
        test_ind = np.where(image==i)
        y_actual = image[test_ind]
        y_predicted = result[test_ind]
        
        
    
        MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 
        
        RMSE.append( math.sqrt(MSE) )
#        
        
        
        MAPE.append( (np.abs((np.subtract(y_actual,y_predicted) / y_actual)).mean())*100)
        
#     np.savetxt(path2+'/CaseA_test_error.csv', [p for p in zip(newelv,RMSE,MAPE)],delimiter=',',fmt ='%s',\
#            header= 'Elv Value, rmse ,  mape ',comments='rmse, mape for test set of 40 interval for case A \n ')    

    return RMSE, MAPE
    


# In[6]:


def compute_jaccardscore(InputImage,cont,newelv):
    Elv = np.unique(InputImage)  #------Calculate no of regions from the original Image-------#
    NoData = Elv[0]
    j_score=[]
    for i in newelv:
        Image1 = np.where(InputImage==i,1,0)
        Image2 = np.where(cont== i, 1 ,0)

        ind = np.where(InputImage > i)
        indxtot = np.where(InputImage > NoData)
        totpixel = len(indxtot[0])

        area = CalPercent(len(ind[0]),totpixel)
        
        set1 = Get_set(InputImage,Image1,i,area)
        set2 = Get_set(InputImage,Image2,i, area)
        j_score.append(Get_JaccardIndex(set1,set2))
        
    return j_score    


# In[20]:


def get_hausdorffdist_pair(InputImage,cont,newelv):
    hd_act=[]
    
    
    hd_pred=[]
    
    
    
    
    contours=[]
    for i in newelv:
        contours.append(i)
        contours.append(i+20)
        
    
    for j in contours:
        
        
        
        
        
        Image1_act = np.where(InputImage==j,1,0)
        Image2_act = np.where(InputImage== j+20, 1 ,0)
        
        
        
        Image1_pred = np.where(cont==j,1,0)
        Image2_pred = np.where(cont == j+20, 1 ,0)
        h1_act = hausdorff_distance(Image1_act, Image2_act, distance='manhattan')
        
        h1_pred = hausdorff_distance(Image1_pred, Image2_pred, distance='manhattan')
        
        
        
        
        hd_act.append(h1_act)
        
        
        hd_pred.append(h1_pred)
        
        
        
        
        
        
        
        
    return hd_act,hd_pred,contours

    
    


# In[23]:





def getEvaluation(InputImage1,InputImage2,fname1,fname2):


    

    def get_plot2_new_AB(ZoneA,ZoneB):
        Label1= 'Zone A'
        Label2 = 'Zone B'


        fig = make_subplots(
            rows=3, cols=3)
    #         subplot_titles=("Root Mean squared Error", "Mean Absolute percentage Error","Jaccard index "))

        fig.add_trace(go.Scatter(x=['C'+str(i)  for i  in range(1,26)], y= ZoneA['rmse'], 
                    mode="markers" ,name= Label1,marker =dict(color="#17becf") ),
                      row=1, col=1 )
        fig.add_trace(go.Scatter(x=['C'+str(i)  for i  in range(1,19)], y= ZoneB['rmse'], 
                                 mode="markers" ,marker = dict(color="#d62728")),
                      row=1, col=1 )

        fig.add_trace(go.Scatter(x=  ['C'+str(i)  for i  in range(1,26)], y= ZoneA ['mape'] ,
                                 mode="markers",marker =dict(color="#17becf")),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=  ['C'+str(i)  for i  in range(1,19)], y= ZoneB ['mape'] ,
                                 mode="markers",marker = dict(color="#d62728")),
                      row=1, col=2)



        fig.add_trace(go.Scatter(x=  ['C'+str(i)  for i  in range(1,26)], y= ZoneA['jac_score' ] ,
                                 mode="markers",marker =dict(color="#17becf")),
                      row=1, col=3)
        fig.add_trace(go.Scatter(x=  ['C'+str(i)  for i  in range(1,19)], y= ZoneB['jac_score' ] ,
                                 name= Label2, mode="markers",marker = dict(color="#d62728")),
                      row=1, col=3)


        fig.add_trace(go.Histogram(x=ZoneA[ 'rmse'],histnorm='percent',
                                   marker = dict(color = "#17becf")),row=2, col=1 )

        fig.add_trace(go.Histogram(x=ZoneA ['mape'],histnorm='percent',
                                   marker = dict(color = "#17becf")),row=2, col=2 )

    #     fig.add_trace(go.Histogram(x=df_cd['HD_euclid' ],name= Label3,histnorm='percent'),row=2, col=1 )
        fig.add_trace(go.Histogram(x=ZoneA['jac_score'],histnorm='percent',
                                   marker = dict(color = "#17becf")),row=2, col=3 )


        fig.add_trace(go.Histogram(x=ZoneB[ 'rmse'],histnorm='percent', marker = dict(color="#d62728")),row=3, col=1 )

        fig.add_trace(go.Histogram(x=ZoneB ['mape'],histnorm='percent',marker = dict(color="#d62728")),row=3, col=2 )

    #     fig.add_trace(go.Histogram(x=df_cd['HD_euclid' ],name= Label3,histnorm='percent'),row=2, col=1 )
        fig.add_trace(go.Histogram(x=ZoneB[ 'jac_score'],histnorm='percent',marker = dict(color="#d62728")),row=3, col=3 )



        for trace in fig['data']:
            if(trace['name'] ==  None):
                trace['showlegend'] = False
        fig.update_layout(

            xaxis1_title_text="Contours ",
            xaxis2_title_text= "Contours ",
            xaxis3_title_text= "Contours",
    #         xaxis4_title_text= "RMSE",
    #         xaxis5_title_text= "MAPE",
    #         xaxis6_title_text= "jaccard index",


            xaxis7_title_text= "RMSE",
            xaxis8_title_text= "MAPE",
            xaxis9_title_text= "jaccard index",

            yaxis1_title_text="RMSE",

            yaxis2_title_text="MAPE",

            yaxis3_title_text="jaccard index",

            yaxis4_ticksuffix = "%",
            yaxis5_ticksuffix = "%",
            yaxis6_ticksuffix = "%",
            yaxis7_ticksuffix = "%",
            yaxis8_ticksuffix = "%",
            yaxis9_ticksuffix = "%",
            font=dict(

            size=11,  # Set the font size here
    #         color="RebeccaPurple"
        ),


            hoverlabel_font_size=15,


            height=600, width=1000,
                          )
        fig.update_yaxes(
            title_standoff = 0)
        fig.update_xaxes(
            title_standoff = 0)


    #     import os

    #     if not os.path.exists(path):
    #         os.mkdir(path)

    #     fig.write_image(path + img_name + ".png")
    #     fig.write_image(path + img_name + ".pdf")

        fig.show(renderer="png")
        return


    def getPlotHd(ZoneA,ZoneB):

        fig = make_subplots(
                rows=1, cols=2)
        #         subplot_titles=(" ", " ", " ", "  "))

            # rmse vs mape
        fig.add_trace(go.Scatter(x= ZoneA['hd_act'], y=ZoneA['hd_pred'] ,name="Zone A",mode="markers" ,marker =dict(color="#17becf")),
                          row=1, col=1 )
        fig.add_trace(go.Scatter(x= ZoneA['hd_act'], y=ZoneA['hd_act'], mode="lines" ,marker = dict(color="#17becf") ),
                          row=1, col=1 )
        fig.add_trace(go.Scatter(x= ZoneB['hd_act'], y=ZoneB['hd_pred'], name ="Zone B",mode="markers" ,marker = dict(color="#d62728") ),
                          row=1, col=2 )
        fig.add_trace(go.Scatter(x= ZoneB['hd_act'], y=ZoneB['hd_act'], mode="lines" ,marker = dict(color="#d62728") ),
                          row=1, col=2 ),
        fig.update_layout( 

                xaxis1_title_text="HD actual",
                yaxis1_title_text=" HD interpolated",
                xaxis2_title_text= "HD actual",
                yaxis2_title_text=" HD interpolated",
                font=dict(

                size=11,  # Set the font size here
        #         color="RebeccaPurple"
            ),

            height=400, width=750,
                              title_text="Pairwise hausdorff distance for both case study "

                         )
        fig.update_yaxes(
                title_standoff = 0)
        fig.update_yaxes(
                title_standoff = 0)

        for trace in fig['data']:
                if(trace['name'] ==  None):
                    trace['showlegend'] = False

    #     import os

    #     if not os.path.exists(path4):
    #         os.mkdir(path4)

#         PlotImage = fig.write_image(path4 + img_name7 + ".png")
#         fig.write_image(path4 + img_name7 + ".pdf")




        fig.show(renderer="png")
        return

    

    def getError(InputImage1,result1,cont_iter1,newelv1,fname):
        rmse1,mape1 = Get_error(InputImage1,result1,newelv1)
        jac_score1= compute_jaccardscore(InputImage1,cont_iter1,newelv1)
        hd_act1,hd_pred1,ContElv1 =   get_hausdorffdist_pair(InputImage1,result1,newelv1)
        
        np.savetxt('./Error_'+ fname+'.csv', [p for p in zip(newelv1,rmse1,mape1,jac_score1)],delimiter=',',fmt ='%s',           header= 'Elv Value, rmse ,  mape ,jaccard_score',comments=' ') 
        np.savetxt('./HausdorffDistance_'+ fname+'.csv', [p for p in zip(hd_act1,hd_pred1,ContElv1)],delimiter=',',fmt ='%s',           header= 'Hd_Act,Hd_pred,Contour',comments=' ') 

        Zone = {'newelv':newelv1,
             'rmse': rmse1,
             'mape' :mape1,
             'jac_score':jac_score1,
                'hd_pred':hd_pred1,
             'hd_act':hd_act1,
             'ContElv1':ContElv1

            }
        return Zone


    def getDisplay(InputImage,TestImage,ResultTest,imgname):

        dpi = matplotlib.rcParams['figure.dpi']
        dspSize  =  1.2* InputImage.shape[1]/float(dpi),1.2*InputImage.shape[0]/float(dpi)

        row,col = 1,3
        fig, ((ax1,ax2,ax3)) = plt.subplots(row,col,figsize = (dspSize[0]*col,dspSize[1]*row))

        ax1.set_title('Contour Set '+ '\n'+imgname)
        ax1.imshow(InputImage,cmap='gray')

        ax2.set_title('Test Contours'+ 'n'+ imgname)
        ax2.imshow(TestImage,cmap ='gray')

        ax3.set_title('Interpolated Surface \n'+ imgname)
        ax3.imshow(ResultTest,cmap ='plasma')

        return




    se =  disk(1)
    perc = 60
    iteration =1


    hillTop1,Boundary1 = getHilBound(fname1)
    hillTop2,Boundary2 = getHilBound(fname2)

    Bseed1, hseed1,helv1 = getSeed(fname1)
    Bseed2, hseed2,helv2 = getSeed(fname2)



    TestImage1,newelv1 = GenTestContours(InputImage1,40,hillTop1,Boundary1)
    workingimg1,Elv,NoData = PreproWithoutHill(TestImage1,Bseed1,hseed1,helv1,se)
    BinaryInput1 = ThrDecomp(TestImage1,workingimg1,perc)
    workingimg_mod1,Mod_InpImg1,BinaryInput1 = Get_hill_set(TestImage1,workingimg1,BinaryInput1,hseed1,helv1,20,se,NoData)
    cont1= IntContour1(Mod_InpImg1,workingimg_mod1,BinaryInput1,se)
    cont_iter1 = IntContourVal(TestImage1,workingimg1,BinaryInput1,se,iteration)
    test = cont1.copy()
    result1 = postprocessing1(test)




    TestImage2,newelv2 = GenTestContours(InputImage2,40,hillTop2,Boundary2)

    workingimg2,Elv,NoData = PreproWithoutHill(TestImage2,Bseed2,hseed2,helv2,se)
    BinaryInput2 = ThrDecomp(TestImage2,workingimg2,perc)
    workingimg_mod2,Mod_InpImg2,BinaryInput2 = Get_hill_set(TestImage2,workingimg2,BinaryInput2,hseed2,helv2,20,se,NoData)
    cont2= IntContour1(Mod_InpImg2,workingimg_mod2,BinaryInput2,se)
    cont_iter2 = IntContourVal(TestImage2,workingimg2,BinaryInput2,se,iteration)

    test = cont2.copy()
    result2 = postprocessing1(test)

    #display

    getDisplay(InputImage1,TestImage1,result1,fname1)
    getDisplay(InputImage2,TestImage2,result2,fname2)


#     print(" Error Analysis for both the test case")

    ZoneA = getError(InputImage1,result1,cont_iter1,newelv1,fname1)
    ZoneB = getError(InputImage2,result2,cont_iter2,newelv2,fname2)
    get_plot2_new_AB(ZoneA,ZoneB)
    getPlotHd(ZoneA,ZoneB)
    return


# In[22]:


def getEvaluateToyData(iteration):
    
    InputImage  = imread('mountain washington 300x300.tif') 
    imgname = 'mountain washington 300x300'
    
    def getFixedInterp(TestImage,iteration,imgname):
        se =  disk(1)
        perc = 60
        
        Bseed, hseed,helv = getSeed(imgname)
        workingimg_tst,Elv_tst ,NoData = PreproWithoutHill(TestImage,Bseed,hseed,helv,se)
        BinaryInput_tst = ThrDecomp(TestImage,workingimg_tst,perc)
        cont_iter_1 = IntContourVal(TestImage,workingimg_tst,BinaryInput_tst,se,iteration=1)
        
        return cont_iter_1
    


        
    hillTop,Boundary = getHilBound(imgname)
    TestImage,newelv = GenTestContours(InputImage,40,hillTop,Boundary)
    
    cont = getFixedInterp(TestImage,iteration,imgname)
    
    
        
    plot = PlotIntActual(InputImage,cont)
    
    dpi = matplotlib.rcParams['figure.dpi']
    dspSize  =  1* InputImage.shape[1]/float(dpi),1*InputImage.shape[0]/float(dpi)

    row,col = 1,3
    fig, ((ax1,ax2,ax3)) = plt.subplots(row,col,figsize = (dspSize[0]*col,dspSize[1]*row))

    ax1.set_title('Original Contour Set ')
    ax1.imshow(InputImage,cmap='gray')

    ax2.set_title('Test Contours')
    ax2.imshow(TestImage,cmap ='gray')

    ax3.set_title('Interpolated contour and ground truth contour \n')
    ax3.imshow(plot.astype('uint8'))

    return
    


# In[ ]:





# In[ ]:




