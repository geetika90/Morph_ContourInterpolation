#!/usr/bin/env python
# coding: utf-8

# In[39]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))

import os
import numpy as np
from skimage.io import imread, imsave,imshow
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation,binary_erosion,dilation,erosion,octagon,opening,closing
from skimage.morphology import closing, disk, erosion
# import cv2
from skimage.color import rgb2gray 
import random
import joblib
from scipy.ndimage import gaussian_filter
import time


# In[40]:


def Regiongrow(image, seed, c):
    row,col = image.shape
#     SeedMarked = np.zeros((row,col))
    SeedMarked = np.where(image > 0,255,0)
    Seedlist =[]
    mark = 255
    c=4
    #define connectivity

    if (c==4):
        neighbour = [(-1,0),(1,0),(0,-1),(0,1)]
    elif(c==8):
        neighbour = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    else:
        print('Error in input:\n please enter 4 for 4-connect and 8 for 8 connect :')


    for seeds in seed:
        Seedlist.append(seeds)



    while (len(Seedlist) > 0 ):


        CurrPoint = Seedlist.pop(0)
        SeedMarked[CurrPoint[0],CurrPoint[1]] = mark
    #         image[CurrPoint[0],CurrPoint[1]] = mark


        for i in range(c):
            New_x = CurrPoint[0] + neighbour[i][0]
            New_y = CurrPoint[1] + neighbour[i][1]


            CheckInside = (New_x < 0) or (New_x >=row) or (New_y < 0) or (New_y >= col)

            if CheckInside:
                continue
            if ((image[New_x,New_y] <= 0)  &  (SeedMarked[New_x,New_y]<=0)):

                SeedMarked [New_x,New_y] = mark
    #                 image[New_x,New_y] = mark

                Seedlist.append((New_x,New_y))
    return SeedMarked


# In[41]:


def get_elev(x):
    val = np.unique(x)
    assert (len(val)==2), "region has multiple value !"
    
    return val[1]
        


# In[42]:


def convert_binary(x):
    x = (x>0)*1
    return(x)


# In[43]:


def GetMedian(x,y,se):
    """
    input: 
        X,Y: 2d Image
        se : structuring element
    Output;
        Med = 2D Image; median set
        
    """
    
#     Area_X = np.sum(X)
#     Area_Y = np.sum(Y)
#     r,c = X.shape
#     Area_tot = r*c
#     if( (CalPercent(Area_X,Area_tot) >50) or  (CalPercent(Area_Y,Area_tot)>50)):
#         print(" complement can be considered")
#         X = 1- X
#         Y = 1-Y
        
        
    se = se
    x_int = np.logical_and(x,y)
    x_uni = np.logical_or(x,y)
    
    
    
    x_dil1 = binary_dilation(x_int,se)
    x_ero1 = binary_erosion(x_uni,se)
    
    x_med1 =  x_dil1 & x_ero1
    
    ## n=2
    x_dil2 = binary_dilation(x_dil1,se)
    x_ero2 = binary_erosion(x_ero1,se)
    
    x_med2 = x_dil2 & x_ero2
    x_med2 = x_med2 | x_med1
    
    n = 3
    
    while(np.logical_xor((x_med1 & x_med2),(x_med1 | x_med2)).any() ): 
    #while(n<20):
        x_dil1 = x_dil2
        x_ero1 = x_ero2
        x_med1 = x_med2
    
        x_dil2 = binary_dilation(x_dil1,se)
        x_ero2 = binary_erosion(x_ero1,se)
        x_med2 = (x_dil2 & x_ero2)
        x_med2 = (x_med2 | x_med1)
    
        
        n= n+1
    
    return x_med2


# In[44]:


def CalPercent(part,whole):
    perc = 100* (part/whole)
    return perc


# In[45]:


def get_contour(X,se):
    se= se
    "get contour/ erosion gradient edge using erosion"
#     X_dil = binary_dilation(X,se)
    X_er = binary_erosion(X,se)
    X_map = np.logical_xor(X_er,X)
    
    return X_map
    
    


# In[46]:


def hillshade(array,azimuth,angle_altitude):
    azimuth = 360.0 - azimuth 

    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi/180.
    altituderad = angle_altitude*np.pi/180.

    shaded = np.sin(altituderad)*np.sin(slope) + np.cos(altituderad)*np.cos(slope)*np.cos((azimuthrad - np.pi/2.) - aspect)

    return 255*(shaded + 1)/2


# In[47]:


def preprocess_fill(image, seed, c, elevation):
    """
    elevation: boolean, 0 or 1
                0: return binary image
                1: return gray image with filled elevation
    """
    row,col = image.shape
    SeedMarked = np.where(image > 0,255,0)

#     SeedMarked = np.zeros((row,col))
    Seedlist =[]
    mark = 1
    elevation=1
    
    #define connectivity
    
    if (c==4):
        neighbour = [(-1,0),(1,0),(0,-1),(0,1)]
    elif(c==8):
        neighbour = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    else:
        print('Error in input:\n please enter 4 for 4-connect and 8 for 8 connect :')
    
    
    for seeds in seed:
        Seedlist.append(seeds[0])
#         print(seeds[0])
        
    
    
        while (len(Seedlist) > 0 ):
        
    
            CurrPoint = Seedlist.pop(0)
            SeedMarked[CurrPoint[0],CurrPoint[1]] = mark
            image[CurrPoint[0],CurrPoint[1]] = seeds[1]
        
            for i in range(c):
                New_x = CurrPoint[0] + neighbour[i][0]
                New_y = CurrPoint[1] + neighbour[i][1]
            
            
                CheckInside = (New_x < 0) or (New_x >=row) or (New_y < 0) or (New_y >= col)
            
                if CheckInside:
                    continue
                if ((image[New_x,New_y] <= 0)  &  (SeedMarked[New_x,New_y]==0)):
                
                    SeedMarked [New_x,New_y] = mark
                    image[New_x,New_y] = seeds[1]
                    Seedlist.append((New_x,New_y))
    
    if (elevation== 0):
            return  SeedMarked 
    else:
            return image
    


# In[48]:


"""
** Preprocessig step:
1. fill all the hill-tp region with the elevation value greater than its correspondig contour elevation value
    Elv(cont) = x
    Elv(hill-top) = x+1
2. fill all the background area which does not belong to contour space.



"""
def preprocessing(InputImage,seed):
    imgcopy = InputImage.copy()
    Elv = np.unique(InputImage)  #------Calculate no of regions from the original Image-------#
    NoData = Elv[0]
    Elv = np.delete(Elv,0)
    No_Reg = len(Elv)
    print("total number of region present are : ",len(Elv),'\n',Elv)
    
#     start_time = time.time()  
    workingimg = preprocess_fill (imgcopy, seed, c=4,elevation=1)
    return workingimg,Elv,NoData


    


# In[49]:


def ThrDecomp(InputImage,workingimg,perc):
    imgcopy=workingimg
    Elv = np.unique(InputImage)  #------Calculate no of regions from the original Image-------#
    NoData = Elv[0]
    Elv = np.delete(Elv,0)
    ElvReg = []
    Seedidx = []
    BinaryInput =[]
    indxtot = np.where(InputImage > NoData)
    totpixel = len(indxtot[0])
    thrperc = perc
#     try: 
# #     os.makedirs("Mt. washington 800/Result/Input")
#         os.makedirs(STpath+ '/Result/Input')
#     except OSError as error: 
#         print(error) 
        
    start_time = time.time()
    
    for i in range(0, len(Elv)-1):
        elv1 = Elv[i]
        elv2 = Elv[i+1]
        ind1 = np.where(InputImage>elv1)
        ind2 = np.where(InputImage >elv2)

        perc1 = CalPercent(len(ind1[0]),totpixel)
        perc2 = CalPercent(len(ind2[0]),totpixel)

        if ((perc1 and perc2) >thrperc):
            if (len(BinaryInput)==0):
                temp1 = np.where(InputImage == elv1,255,0)
                temp2 = np.where(InputImage == elv2,255, 0)
    #             print(elv1,elv2)
                seed1 = np.where(np.logical_and(imgcopy < elv1,imgcopy > NoData))
                seed2 = np.where(np.logical_and(imgcopy < elv2,imgcopy > NoData))


                seedlist1 = [(seed1[0][i],seed1[1][i]) for i in range(len(seed1[0]))  ] 
                seedlist2 = [(seed2[0][i],seed2[1][i]) for i in range(len(seed2[0])) ] 
                TempInput1 =  Regiongrow(temp1, seedlist1, c=4)

                TempInput2 =  Regiongrow(temp2, seedlist2, c=4)
                BinaryInput.append((TempInput1,TempInput2,elv1,elv2))
#                 plt.imsave(STpath+'/Result/Input/binput'+ str(i)+str(elv1)+'.png',TempInput1)
#                 plt.imsave(STpath +  '/Result/Input/binput'+str(i)+str(elv2)+'.png',TempInput2)
#     #             plt.imsave('Mt. washington 800/Result/Input/binput'+ str(i)+str(elv1)+'.png',TempInput1)
    #             plt.imsave('Mt. washington 800/Result/Input/binput'+str(i)+str(elv2)+'.png',TempInput2)


    #             print(perc1,perc2,elv1,elv2,i)

            else:
                TempInput1 = BinaryInput[i-1][1]
                temp2 = np.where(InputImage == elv2,255, 0)
                seed2 = np.where(np.logical_and(imgcopy < elv2,imgcopy > NoData))
                seedlist2 = [(seed2[0][i],seed2[1][i]) for i in range(len(seed2[0])) ] 
                TempInput2 =  Regiongrow(temp2, seedlist2, c=4)
                BinaryInput.append((TempInput1,TempInput2,elv1,elv2))
#                 plt.imsave(STpath+'/Result/Input/binput'+ str(i)+str(elv1)+'.png',TempInput1)
#                 plt.imsave(STpath +  '/Result/Input/binput'+str(i)+str(elv2)+'.png',TempInput2)
    #             plt.imsave('Mt. washington 800/Result/Input/binput'+ str(i)+str(elv1)+'.png',TempInput1)
    #             plt.imsave('Mt. washington 800/Result/Input/binput'+str(i)+str(elv2)+'.png',TempInput2)

    #             print(perc1,perc2, elv1,elv2,i)

        elif((perc1>thrperc and perc2<thrperc) or (perc1<thrperc and perc2>thrperc)):
            if (len(BinaryInput)==0):
                temp1 = np.where(InputImage == elv1,255,0)
                temp2 = np.where(InputImage == elv2,255, 0)
    #             print(elv1,elv2)
                seed1 = np.where(imgcopy>elv1)
                seed2 = np.where(imgcopy>elv2)


                seedlist1 = [(seed1[0][i],seed1[1][i]) for i in range(len(seed1[0]))  ] 
                seedlist2 = [(seed2[0][i],seed2[1][i]) for i in range(len(seed2[0])) ] 
                TempInput1 =  Regiongrow(temp1, seedlist1, c=4)

                TempInput2 =  Regiongrow(temp2, seedlist2, c=4)
                BinaryInput.append((TempInput1,TempInput2,elv1,elv2))
#                 plt.imsave(STpath+'/Result/Input/binput'+ str(i)+str(elv1)+'.png',TempInput1)
#                 plt.imsave(STpath +  '/Result/Input/binput'+str(i)+str(elv2)+'.png',TempInput2)
#     #             plt.imsave('Mt. washington 800/Result/Input/binput'+ str(i)+str(elv1)+'.png',TempInput1)
    #             plt.imsave('Mt. washington 800/Result/Input/binput'+str(i)+str(elv2)+'.png',TempInput2)

    #             print(elv1,elv2,i)

            else:
                temp1 = np.where(InputImage == elv1,255,0)
                temp2 = np.where(InputImage == elv2,255, 0)
        #             print(elv1,elv2)
                seed1 = np.where(imgcopy>elv1)
                seed2 = np.where(imgcopy>elv2)


                seedlist1 = [(seed1[0][i],seed1[1][i]) for i in range(len(seed1[0]))  ] 
                seedlist2 = [(seed2[0][i],seed2[1][i]) for i in range(len(seed2[0])) ] 
                TempInput1 =  Regiongrow(temp1, seedlist1, c=4)

                TempInput2 =  Regiongrow(temp2, seedlist2, c=4)
                BinaryInput.append((TempInput1,TempInput2,elv1,elv2))
#                 plt.imsave(STpath+'/Result/Input/binput'+ str(i)+str(elv1)+'.png',TempInput1)
#                 plt.imsave(STpath +  '/Result/Input/binput'+str(i)+str(elv2)+'.png',TempInput2)

    #             plt.imsave('Mt. washington 800/Result/Input/binput'+ str(i)+str(elv1)+'.png',TempInput1)
    #             plt.imsave('Mt. washington 800/Result/Input/binput'+str(i)+str(elv2)+'.png',TempInput2)

    #             print(perc1,perc2,elv1,elv2,i)
    #             print('hihi')

        elif(perc1<thrperc and perc2<thrperc ):
            if(len(BinaryInput)==0):
                temp1 = np.where(InputImage == elv1,255,0)
                temp2 = np.where(InputImage == elv2,255, 0)
                seed1 = np.where(imgcopy > elv1)
                seed2 = np.where(imgcopy > elv2)
                seedlist1 = [(seed1[0][i],seed1[1][i]) for i in range(len(seed1[0]))  ] 
                seedlist2 = [(seed2[0][i],seed2[1][i]) for i in range(len(seed2[0])) ] 
                TempInput1 =  Regiongrow(temp1, seedlist1, c=4)
                TempInput2 =  Regiongrow(temp2, seedlist2, c=4)
                BinaryInput.append((TempInput1,TempInput2,elv1,elv2))
#                 plt.imsave(STpath+'/Result/Input/binput'+ str(i)+str(elv1)+'.png',TempInput1)
#                 plt.imsave(STpath +  '/Result/Input/binput'+str(i)+str(elv2)+'.png',TempInput2)
              
            else:




                TempInput1 = BinaryInput[i-1][1]
                temp2 = np.where(InputImage == elv2,255, 0)
                seed2 = np.where(imgcopy>elv2 )
                seedlist2 = [(seed2[0][i],seed2[1][i]) for i in range(len(seed2[0])) ] 
                TempInput2 =  Regiongrow(temp2, seedlist2, c=4)
                BinaryInput.append((TempInput1,TempInput2,elv1,elv2))
#                 plt.imsave(STpath+'/Result/Input/binput'+ str(i)+str(elv1)+'.png',TempInput1)
#                 plt.imsave(STpath +  '/Result/Input/binput'+str(i)+str(elv2)+'.png',TempInput2)

#     print("--- %s seconds ---" % (time.time() - start_time))   
    return BinaryInput


# In[50]:


def IntContour(InputImage,WorkinImg,BinaryInput,se,path):
    
    ContBinary = np.where(InputImage >0, 1,0)
    
    cont = WorkinImg.copy()
    n= 0

    start_time = time.time()

    for i in BinaryInput:
        Inpt_X = i[0]
        Inpt_Y = i[1]
        Elv1 = i[2]
        Elv2 = i[3]

        queue = []
        queue.append(((Inpt_X,Elv1),(Inpt_Y,Elv2)))
        InterSect = np.logical_and(Inpt_X,Inpt_Y)
        Union = np.logical_or(Inpt_X,Inpt_Y)
        ContSpace = np.logical_xor(InterSect,Union)
        indxS = np.where(ContSpace > 0)
        PrevArea = 0
        Area = np.sum(ContBinary[indxS])

        while( Area != PrevArea):

            for ii in range(len(queue)):
                inpt = queue.pop(0)
                X = inpt[0][0]
                Y = inpt[1][0] 
                height1 = inpt[0][1]
                height2 = inpt[1][1]
                me = GetMedian(X,Y,se)
                me_cont = get_contour(me,se)  #-------- me_cont : Binary Image of generated intermediate contour ------------#
                height_me = (height1+height2)/2  #-----elevation of the intermediate contour -----#
    #             me_cont = (me_cont>0)*height_me    #----- Gray Scale Image ----#
    #             me = (me > 0 ) * I_me 

                indx_m = np.where(np.logical_and(me_cont > 0,cont <= 0))
                cont[indx_m] = height_me
                ContBinary = convert_binary(cont)

                l_inpt = ((Inpt_X,Elv1),(me,height_me))
                r_inpt =((me,height_me),(Inpt_Y,Elv2))
                queue.append(l_inpt)
                queue.append(r_inpt)


                try: 
                    os.makedirs(path+'/Result/SrcTrgt_Result'+ str(n+1))
                except OSError as error: 
                    print(error) 



    #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/Input1'  + str(i+1) + '.tiff' ,X, dpi= 500 , cmap='gray')  
    #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/Input2' + str(i+1) + '.tiff' ,Y, dpi= 500 , cmap='gray')  
    #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/MER'    + str(i+1) + '.tiff' ,me, dpi= 500 , cmap='gray')  
    #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/Intcont' + str(i+1) + '.tiff' ,cont, dpi= 500 , cmap='gray')  
    #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/Intcont_binary' + str(i+1) + '.tiff' ,ContBinary   , dpi= 500 , cmap='gray')  



            PrevArea = Area
            Area = PrevArea+ len(indx_m[0])
#           print(PrevArea,Area)








    #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/Input1'  + str(ii+1) + '.tiff',X,cmap='gray')
    #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/Input2'  + str(ii+1) + '.tiff',Y,cmap='gray')
    #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/MER'  + str(ii+1) + '.tiff',me,cmap='gray')
    #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/Intcont'  + str(ii+1) + '.tiff',cont,cmap='gray')
    #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/Intcont_binary'  + str(ii+1) + '.tiff',ContBinary,cmap='gray')



        plt.imsave(path+ '/Result/SrcTrgt_Result' + str(n+1)+'/Input1'   + '.tiff',Inpt_X,cmap='gray')
        plt.imsave(path + '/Result/SrcTrgt_Result' + str(n+1)+'/Input2'   + '.tiff',Inpt_Y,cmap='gray')
        # plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/MER'  + str(ii+1) + '.tiff',me,cmap='gray')
        plt.imsave(path + '/Result/SrcTrgt_Result' +str(n+1)+'/Intcont'  + str(ii+1) + '.tiff',cont,cmap='gray')
        plt.imsave(path + '/Result/SrcTrgt_Result' +  str(n+1)+'/Intcont_binary'  + str(ii+1) + '.tiff',ContBinary,cmap='gray')

        n= n+1




    joblib.dump(cont,path+'/Result/Intcontour'+'.z')

    extime = time.time() - start_time
    print("--- %s seconds ---" % (extime))
    return cont


# In[ ]:





# In[51]:


def IntContour1(InputImage,WorkinImg,BinaryInput,se):
    
    ContBinary = np.where(InputImage >0, 1,0)
#     plt.imshow(ContBinary)
#     plt.show()
    ii = 0
    cont = WorkinImg.copy()
    n= 0
    p =0
    import time
#     start_time = time.time()

    for i in BinaryInput:

#         try: 
#             os.makedirs(path+ '/Result/SrcTrgt_Result'+str(p+1))
#         except OSError as error: 
#             print(error) 

        Inpt_X = i[0]
        Inpt_Y = i[1]
        Elv1 = i[2]
        Elv2 = i[3]

        queue = []
        queue.append(((Inpt_X,Elv1),(Inpt_Y,Elv2)))
        
        ContSpace = findContSpace(Inpt_X,Inpt_Y)
        
        
        indxS = np.where(ContSpace > 0)
        ContSpace[indxS] = 255
        
        PrevArea = 0
        Area = np.sum(ContBinary[indxS])
#         print(PrevArea,Area)

        while( Area != PrevArea):
            pcount = 0

#             print('-----Loop Running-----Number==',n)

            while (len(queue) != 0):
                ccount = 0


#                 print('lenth of the queue=', len(queue))

                inpt = queue.pop(0)

                X = inpt[0][0]
                Y = inpt[1][0] 
                
                contspace = findContSpace(X,Y)
                
                
                height1 = inpt[0][1]
                height2 = inpt[1][1]
                me = GetMedian(X,Y,se)
#                 me = opening(me,se)

                me_cont = get_contour(me,se)  #-------- me_cont : Binary Image of generated intermediate contour ------------#
    #               #-----elevation of the intermediate contour -----#
    #             me_cont = (me_cont>0)*height_me    #----- Gray Scale Image ----#
    #             me = (me > 0 ) * I_me 
    #             print(height_me)
                
                ind_c = np.where(contspace > 0)

                indx_m = np.where(np.logical_and(me_cont > 0,ContSpace>0 ,contspace>0))

                pcount= ccount
                ccount = pcount+ len(indx_m[0])

                if (ccount == pcount):
#                     print('there is no new point to add---',pcount,ccount)
                    continue

                height_me = (height1+height2)/2
                cont[indx_m] = height_me
                ContSpace[indx_m]=0
                
    #             ContBinary = convert_binary(cont)


#                 print('New point added',pcount,ccount)

                l_inpt = ((Inpt_X,Elv1),(me,height_me))
                r_inpt =((me,height_me),(Inpt_Y,Elv2))

                queue.append(l_inpt)
                queue.append(r_inpt)


            PrevArea = Area
            Area = PrevArea+ len(indx_m[0])
#             print('total Area assigned',PrevArea,Area)





    #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/Input1'  + str(i+1) + '.tiff' ,X, dpi= 500 , cmap='gray')  
    #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/Input2' + str(i+1) + '.tiff' ,Y, dpi= 500 , cmap='gray')  
    #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/MER'    + str(i+1) + '.tiff' ,me, dpi= 500 , cmap='gray')  
    #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/Intcont' + str(i+1) + '.tiff' ,cont, dpi= 500 , cmap='gray')  
    #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/Intcont_binary' + str(i+1) + '.tiff' ,ContBinary   , dpi= 500 , cmap='gray')  




    #         print(PrevArea,Area)



#             n= n+1




    #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/Input1'  + str(ii+1) + '.tiff',X,cmap='gray')
    #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/Input2'  + str(ii+1) + '.tiff',Y,cmap='gray')
    #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/MER'  + str(ii+1) + '.tiff',me,cmap='gray')
    #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/Intcont'  + str(ii+1) + '.tiff',cont,cmap='gray')
    #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/Intcont_binary'  + str(ii+1) + '.tiff',ContBinary,cmap='gray')



#         plt.imsave(path+ '/Result/SrcTrgt_Result' + str(p+1)+'/Input1'   + '.tiff',Inpt_X,cmap='gray')
#         plt.imsave(path + '/Result/SrcTrgt_Result' + str(p+1)+'/Input2'   + '.tiff',Inpt_Y,cmap='gray')
#         # plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/MER'  + str(ii+1) + '.tiff',me,cmap='gray')
#         plt.imsave(path + '/Result/SrcTrgt_Result' +str(p+1)+'/Intcont'  + str(ii+1) + '.tiff',cont,cmap='gray')
#     #     plt.imsave(path + '/Result/SrcTrgt_Result' +  str(n+1)+'/Intcont_binary'  + str(ii+1) + '.tiff',ContBinary,cmap='gray')
        p =p+1





#     joblib.dump(cont,path+'/Result/Intcontour'+'.z')

#     extime = time.time() - start_time
#     print("--- %s seconds ---" % (extime))
    return cont


# In[52]:


def postprocessing1(Output):
#-------- Postprocessing steps: for all the unfilled pixels, ------------#
#---------replace with the average of their neighbour area elevation value -----------#
    cont = Output
    r,c = cont.shape
    
    indx = np.where(cont<=0)
    
    
    kernel = [(-1,0),(1,0),(0,-1),(0,1)]
    while(len(indx[0])!= 0):
        indx = np.where(cont<=0)
        
#         r = len(indx[0])
#         c = len(indx[1])
#         print('the lenth of the array-',r,c)
        for i in indx[0]:
            for j in indx[1]:
                pval = 0
                count = 0
                if(cont[i][j] > 0):
                    continue
                else:
                    p = (i,j)
                    for ii in range(4):
                        n = (p[0]+ kernel[ii][0],p[1]+ kernel[ii][1])

                        CheckInside = (n[0]<0 or n[1] <0 or n[0]>= r or n[1] >= c)
                        if CheckInside :

                            continue
                        elif (cont[n] >0):
                            pval = pval + cont[n]
                            count = count+1

                    cont[p] = (pval/count)
#         indx = np.where(cont<=0)
#     plt.imsave(path + '/Result/Intcont_final'   + '.tiff',cont,cmap='gray')
#     joblib.dump(cont,path + '/Result/Intcont_final'+'.z')  
    return cont


# In[53]:


def findContSpace(x,y):
    
    InterSect = np.logical_and(x,y)
    Union = np.logical_or(x,y)

    ContSpace = np.logical_xor(InterSect,Union)
    
    return ContSpace


# In[54]:


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





# In[55]:


def GenTestContours(InputImage,ContIntvl,hillTop,Boundary):
    TestImage = InputImage.copy()
    elv = np.unique(InputImage)
    nodata = elv[0]
    elv = np.delete(elv,0)
    ElvSkip =[]
    
    
    if (ContIntvl==40):
        newelv= elv[1:len(elv)-2:2]
    for i in newelv:

        if (i not  in Boundary) and (i not  in hillTop):
            


            tmp = np.where(TestImage==i)
        #         val = InputImage[tmp]
            TestImage[tmp]= nodata
            ElvSkip.append(i)

        else:
            newelv = [newelv != i]


    return TestImage,ElvSkip
    


# In[56]:


def Get_hill_set(Image,workingimg,SetInpt,hseed,helv,CInterval,se,NoData):
    InputImage = Image.copy()
    WorkImg = workingimg.copy()
    
    
    for i in range(len(hseed)):
    
        elv1 = helv[i]
        elv2 = elv1+ CInterval
        temp = np.where(InputImage== helv[i],255,0)
        temphill = Regiongrow(temp,[hseed[i]],c=4)
        space = np.where(np.logical_xor(np.logical_and(temp, temphill),np.logical_or(temp,temphill)))
        WorkImg[space] = NoData
    
        
        er1= temphill
        jj= 0
        while(er1.any()):
        
            er1 = binary_erosion(er1,se)
            jj =jj+1
#             print(jj)
        er1= temphill
        for i in range(jj-2):
            er1 = binary_erosion(er1)
        
        SetInpt.append((temphill,er1,elv1,elv2))
        
        indx1 = np.where(er1>0)
        c1 = get_contour(er1,se)
        indx2 = np.where(c1>0)

        WorkImg[indx1]= elv2+1
        InputImage[indx2]=elv2
        
        


    return WorkImg,InputImage,SetInpt


# In[80]:


def getSeed(ImgName):
    
        if ImgName=='mountain washington 800x800':
            Bseed = [((357,2),2738),((779,5),2298),((792,795),2538),((596,797),2579),((2,754),2298),((2,2),2739),]
            hseed = [(149,174),(577,449),(297,630),(411,268)]
            helv= [3540,3340,3200,3440]
            return Bseed,hseed,helv
            
        if ImgName == 'mountain washington 500x500':
            Bseed =[((493,25),1139),((305,12),1139),((439,2),1139),((0,499),1920),((250,491),1481)]
            hseed=[]
            helv=[]
            return Bseed,hseed,helv
            
        if ImgName == 'mountain washington 300x300':
            hseed = [(173,102), (3,163)]
            helv= [3200,3220]
            Bseed= [((173,102),3202),((280,2),2938),((292,91),3062),((66,1),2897),((3,163),3221),((257,297),2779)]

            
            return Bseed,hseed,helv
        if ImgName == 'Toy image':
            Bseed =[]
            helv=[100]
            hseed = [(227,325)]
            return Bseed,hseed,helv
            
            
        


# In[79]:


def getHilBound(imgname):
    if imgname== 'mountain washington 500x500':
        hillTop =[]
        Boundary = [1140,1920,1480]
        return hillTop,Boundary 
    if imgname == 'mountain washington 800x800':
        hillTop = [3540,3200,3440,3340]
        Boundary = [2738,2540,2580,2300,2740,2298,2738]
        return hillTop,Boundary 
    if imgname == 'mountain washington 300x300':
        hillTop= [3540,3200, 3440,3340]
        Boundary=[2940,3060,2900,3220,2780]
        
    
        return hillTop,Boundary 


# In[76]:


def Get_set(workingimg , Contour,elv, perc):
    if (perc > 60):
        seed = np.where(np.logical_and(workingimg < elv,workingimg > 0))
        seedlist = [(seed[0][i],seed[1][i]) for i in range(len(seed[0]))  ]
        Set =  Regiongrow(Contour, seedlist, c=4)
        
    else:
        seed = np.where(workingimg > elv)
        seedlist = [(seed[0][i],seed[1][i]) for i in range(len(seed[0]))  ]
        Set =  Regiongrow(Contour, seedlist, c=4)
        
    return Set
    


# In[ ]:




