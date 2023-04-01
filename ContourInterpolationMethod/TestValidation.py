#!/usr/bin/env python
# coding: utf-8

# In[53]:


from ContourInterpolationMethod.All_modules_redifened import *
import cv2
from hausdorff import hausdorff_distance
from skimage import metrics


# In[ ]:


def CompTest(InputImage,ContIntvl):
#----------- Generate the test case of 40m contour interval by skipping alternate elevation ----------------#
# InputImage = imread('Data/MT.washington cont/mountain washington 300x300.tif')
    TestImage = InputImage.copy()
    elv = np.unique(InputImage)
    nodata = elv[0]
    elv = np.delete(elv,0)
    
    if (ContIntvl==40):
        newelv= elv[1:len(elv)-2:2]
    #     ActVal = []
    #     predVal =[]
        for i in newelv:
            tmp = np.where(TestImage==i)
    #         val = InputImage[tmp]
            TestImage[tmp]= nodata
    #         ActVal.append(val)
    #         predVal.append(val)
    
    if (ContIntvl==80):
        newelv=[]
        for i in range(1,len(elv)-1):
            if(i%4 !=0):
                newelv.append(elv[i])
            continue
        for i in newelv:
            tmp = np.where(TestImage==i)
    #         val = InputImage[tmp]
            TestImage[tmp]= nodata
    #         ActVal.append(val)
    #         predVal.append(val)
        
#     ActVal= np.concatenate(ActVal)
#     predVal= np.concatenate(predVal)

#     from sklearn.metrics import mean_squared_error
#     from math import sqrt
#     rms = sqrt(mean_squared_error(ActVal, predVal))
    return TestImage,newelv


# In[ ]:


def SavePlot(InputImageTest):

    plt.axis('off')
#     plt.figure(figsize = (10,10))

    spec = plt.imshow(InputImageTest)

    plt.savefig('spec',bbox_inches='tight',transparent=True, pad_inches=0)


# In[ ]:





# In[55]:


def PlotIntActual(InputImage,IntCont):
    InputImage2=np.where(InputImage<0,0,1)

    ResultValid2=np.where(IntCont<0,0,1)

    mergeImage=InputImage2+ResultValid2*2
    mergeImage=np.where(mergeImage>2,2,mergeImage)
    colorImage=np.zeros((mergeImage.shape[0],mergeImage.shape[1],3))
    color=np.array([[255,255,255],[0,0,255],[255,0,0]])
    for i in range(colorImage.shape[0]):
        for j in range(colorImage.shape[1]):
            colorImage[i,j]=color[mergeImage[i,j]]
    # cv2.imwrite('Mt. washington 300 Test/Result/colorImage1.png',colorImage)
#     cv2.imwrite(path +'/Result/colorImage2.png',colorImage)
    return colorImage


# In[ ]:


def IntContTest1(InputImage,WorkinImg,BinaryInput,se,pathiteration):
    ContBinary = np.where(InputImage >0, 1,0)
#     plt.imshow(ContBinary)
#     plt.show()
    ii = 0
    cont = WorkinImg.copy()
    n = 0
    p =0
    import time
    start_time = time.time()

    for i in BinaryInput:


        Inpt_X = i[0]
        Inpt_Y = i[1]
        Elv1 = i[2]
        Elv2 = i[3]

        queue = []
        queue.append(((Inpt_X,Elv1),(Inpt_Y,Elv2)))
        
        ContSpace = findContSpace(Inpt_X,Inpt_Y)
        
        
        indxS = np.where(ContSpace > 0)
        ContSpace[indxS] = 255
        
#         PrevArea = 0
#         Area = np.sum(ContBinary[indxS])
#         print(PrevArea,Area)

        while(n < iteration):
            

#             print('-----Loop Running-----Number==',n)

            while (len(queue) != 0):
                ccount = 0


                print('lenth of the queue=', len(queue))

                inpt = queue.pop(0)

                X = inpt[0][0]
                Y = inpt[1][0] 
                
                contspace = findContSpace(X,Y)
                
                
                height1 = inpt[0][1]
                height2 = inpt[1][1]
                me = GetMedian(X,Y,se)

                me_cont = get_contour(me,se)  #-------- me_cont : Binary Image of generated intermediate contour ------------#
    #               #-----elevation of the intermediate contour -----#
    #             me_cont = (me_cont>0)*height_me    #----- Gray Scale Image ----#
    #             me = (me > 0 ) * I_me 
    #             print(height_me)
                
                ind_c = np.where(contspace > 0)

                indx_m = np.where(np.logical_and(me_cont > 0,ContSpace>0 ,contspace>0))

                

                

                height_me = (height1+height2)/2
                cont[indx_m] = height_me
                ContSpace[indx_m]=0
                
    #             ContBinary = convert_binary(cont)


           

                l_inpt = ((Inpt_X,Elv1),(me,height_me))
                r_inpt =((me,height_me),(Inpt_Y,Elv2))

                queue.append(l_inpt)
                queue.append(r_inpt)







    #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/Input1'  + str(i+1) + '.tiff' ,X, dpi= 500 , cmap='gray')  
    #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/Input2' + str(i+1) + '.tiff' ,Y, dpi= 500 , cmap='gray')  
    #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/MER'    + str(i+1) + '.tiff' ,me, dpi= 500 , cmap='gray')  
    #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/Intcont' + str(i+1) + '.tiff' ,cont, dpi= 500 , cmap='gray')  
    #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/Intcont_binary' + str(i+1) + '.tiff' ,ContBinary   , dpi= 500 , cmap='gray')  




    #         print(PrevArea,Area)



            n= n+1




    #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/Input1'  + str(ii+1) + '.tiff',X,cmap='gray')
    #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/Input2'  + str(ii+1) + '.tiff',Y,cmap='gray')
    #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/MER'  + str(ii+1) + '.tiff',me,cmap='gray')
    #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/Intcont'  + str(ii+1) + '.tiff',cont,cmap='gray')
    #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/Intcont_binary'  + str(ii+1) + '.tiff',ContBinary,cmap='gray')



#         plt.imsave(path+ '/Result/SrcTrgt_Result' + str(p+1)+'/Input1'   + '.tiff',Inpt_X,cmap='gray')
#         plt.imsave(path + '/Result/SrcTrgt_Result' + str(p+1)+'/Input2'   + '.tiff',Inpt_Y,cmap='gray')
#         # plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/MER'  + str(ii+1) + '.tiff',me,cmap='gray')
#         plt.imsave(path + '/Result/SrcTrgt_Result' +str(p+1)+'/Intcont'  + str(ii+1) + '.tiff',cont,cmap='gray')
    #     plt.imsave(path + '/Result/SrcTrgt_Result' +  str(n+1)+'/Intcont_binary'  + str(ii+1) + '.tiff',ContBinary,cmap='gray')
#         p =p+1





    joblib.dump(cont,path+'/Result/Test1/Intcontour'+'.z')
    plt.imsave(path  + '/Result/Test1/Intcontour.tiff',cont,cmap='gray')

    extime = time.time() - start_time
    print("--- %s seconds ---" % (extime))
    return cont


# In[57]:


def IntContourVal(InputImage,WorkinImg,BinaryInput,se,iteration):
    
    n =0
    ContList=[]
    while(n<iteration):
        ContBinary = np.where(InputImage >0, 1,0)
    
        cont = InputImage.copy()
    

#     start_time = time.time()
    
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
            k= 0
            while( k <= n ):

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


    #                 try: 
    #                     os.makedirs(path+ '/Result/SrcTrgt_Result'+str(n+1))
    #                 except OSError as error: 
    #                     print(error) 



        #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/Input1'  + str(i+1) + '.tiff' ,X, dpi= 500 , cmap='gray')  
        #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/Input2' + str(i+1) + '.tiff' ,Y, dpi= 500 , cmap='gray')  
        #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/MER'    + str(i+1) + '.tiff' ,me, dpi= 500 , cmap='gray')  
        #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/Intcont' + str(i+1) + '.tiff' ,cont, dpi= 500 , cmap='gray')  
        #             plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/Intcont_binary' + str(i+1) + '.tiff' ,ContBinary   , dpi= 500 , cmap='gray')  



    #             PrevArea = Area
    #             Area = PrevArea+ len(indx_m[0])
    #             print(PrevArea,Area)










        #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/Input1'  + str(ii+1) + '.tiff',X,cmap='gray')
        #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/Input2'  + str(ii+1) + '.tiff',Y,cmap='gray')
        #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/MER'  + str(ii+1) + '.tiff',me,cmap='gray')
        #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/Intcont'  + str(ii+1) + '.tiff',cont,cmap='gray')
        #     plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)++'/Intcont_binary'  + str(ii+1) + '.tiff',ContBinary,cmap='gray')



    #             plt.imsave(path+ '/Result/SrcTrgt_Result' + str(n+1)+'/Input1'   + '.tiff',Inpt_X,cmap='gray')
    #             plt.imsave(path + '/Result/SrcTrgt_Result' + str(n+1)+'/Input2'   + '.tiff',Inpt_Y,cmap='gray')
    #             # plt.imsave('Mt. washington 300/Result/SrcTrgt_Result'+str(n+1)+'/MER'  + str(ii+1) + '.tiff',me,cmap='gray')
    #             plt.imsave(path + '/Result/SrcTrgt_Result' +str(n+1)+'/Intcont'  + str(ii+1) + '.tiff',cont,cmap='gray')
    #             plt.imsave(path + '/Result/SrcTrgt_Result' +  str(n+1)+'/Intcont_binary'  + str(ii+1) + '.tiff',ContBinary,cmap='gray')

                k= k+1
        
        ContList.append(cont)

        n=n+1



#     joblib.dump(cont,path+'/Result/Test1/Intcontour'+'.z')
#     plt.imsave(path  + '/Result/Test1/Intcontour.tiff',cont,cmap='gray')
    
#     extime = time.time() - start_time
#     print("--- %s seconds ---" % (extime))
    if (iteration>1):
        return ContList
    else:
        return cont


# In[54]:


def Get_JaccardIndex(Image1,Image2):
    intersect = np.logical_and(Image1,Image2)
    Union = np.logical_or(Image1,Image2)
    size_inter = len(np.where(intersect>0)[0])
    size_uni = len(np.where(Union>0)[0])
    J_index = (size_inter/size_uni)
    
    return J_index
     
    
    
    


# In[ ]:



    
    


# In[ ]:


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
        
      

    return RMSE, MAPE
    


# In[ ]:


def get_hausdorffdist(InputImage,cont,newelv):
    hd_manhattan=[]
    hd_euclid=[]
    hd_sklearn=[]
    
    for i in newelv:
        Image1 = np.where(InputImage==i,1,0)
        Image2 = np.where(cont== i, 1 ,0)
        
        hd_manhattan.append(hausdorff_distance(Image1, Image2, distance='manhattan'))
        hd_euclid.append(hausdorff_distance(Image1, Image2, distance='euclidean'))
        hd_sklearn.append(metrics.hausdorff_distance(Image1, Image2))
        
        
    return hd_manhattan,hd_euclid,hd_sklearn

    
    


# In[ ]:





# In[ ]:




