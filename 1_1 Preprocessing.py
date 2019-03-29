# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 03:37:50 2018

@author: Sunanda, Ibtihel, Yimeng
"""
# ---------------------------------------
#    Importing the Modules and Libraries
#----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import cv2, os

# ---------------------------------------
#             Defining Variables
#----------------------------------------
process_training_data = True
process_test_data = True
with_PCA = True

# False if you are just debugging your preprocessing
save_files = True

# Save Images while preprocessing? Keep it to False for anyone except the Admin
save_images = False
save_raw_images = False
image_format = ".png"

# Output format
sqr_size = 28

# Global Variables
training_set_size = 50000
test_set_size = 10000
save_to = "LargestDigit"

# Want to use blur?
blur = False
blur_val = 1

# ---------------------------------------
#             DownLoad Data
#----------------------------------------
y = np.loadtxt("http://cs.mcgill.ca/~ksinha4/datasets/kaggle/train_y.csv", delimiter=",")
temp = np.loadtxt("http://cs.mcgill.ca/~ksinha4/datasets/kaggle/train_x.csv", delimiter=",")
temp = np.loadtxt("http://cs.mcgill.ca/~ksinha4/datasets/kaggle/test_x.csv", delimiter=",")

# ---------------------------------------
#        USER DEFINED FUNCTIONS
#----------------------------------------

"""
This function resizes the image while maintaining the aspect ratio.

Credits - Alexander Reynolds [https://stackoverflow.com/users/5087436/alexander-reynolds]
Stachoverflow post - https://stackoverflow.com/questions/44720580/resize-image-canvas-to-maintain-square-aspect-ratio-in-python-opencv

@params img 2D vexctor of the image, size resizing parameter, padcolor 0 for black
@returns scaled_img resized image without distorting aspect ratio
"""
def resizeAndPad(img, size, padColor=0):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img


# This function crops the digit out of the image, 

def get_digit(img,rect):
    # crop the minAreaRectangle image
    (rows, cols) = img.shape
    center = (rows / 2, cols / 2)
    # move of the minAreaRectangle to the center of img
    v_col=32-int(rect[0][0])    #v_col x axis move
    v_row=32-int(rect[0][1])    #v_col y axis move
    M1 = np.float32([[1, 0, v_col], [0, 1, v_row]])
    img = cv2.warpAffine(img, M1, (cols, rows))

    #rotate of the image
    angle=rect[2]
    if rect[1][0]>rect[1][1]:
        angle=(90+angle)
    else:
        angle=angle

    M=cv2.getRotationMatrix2D(center,angle,1.0)
    rot_img=cv2.warpAffine(img,M,(cols,rows),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_TRANSPARENT)
#    cv2.imshow('img3', rot_img)   #show the digit with largest encompassed area

    # rotate bounding box
    box = cv2.boxPoints(rect)
    box1=cv2.transform(np.array([box]), M1)[0]
    pts = np.int0(cv2.transform(np.array([box1]), M))[0]

    # crop
    img_crop = rot_img[pts[2][1]:pts[0][1],
                       pts[1][0]:pts[3][0]]
    
    new_image = resizeAndPad(img_crop, (sqr_size,sqr_size), 0)
#    cv2.imshow('Rotated and cropped', new_image) 
    
    return new_image        


# ---------------------------------------
#           TRAINING DATA
#----------------------------------------
if process_training_data :
    print("Processing the Training Data")
    i=0
    x=np.zeros((training_set_size,sqr_size,sqr_size))
    with open('cs.mcgill.ca/~ksinha4/datasets/kaggle/train_x.csv') as f:
        for line in f: 
            img=np.array(line.split(','))
            img=img.astype(np.float)
            img=img.astype(np.uint8)
            img=np.uint8(img)
            gray=img.reshape(64, 64)
            if save_raw_images:
                save_image_loc = "RawImages/Train/"+ str(int(i)) + "_L"  + str(int(y[i])) + image_format
                plt.imsave(save_image_loc, gray, cmap="gray")
            gray[gray<255] = 0
            thresh = gray
            _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # get contours
        
            area0 = 0
            # for each contour found, draw a rectangle around it on original image
            for cont in contours:           #get the contour and minAreaRectangle with the largest area
                rect1 = cv2.minAreaRect(cont)  # cv2.minAreaRect() returns: (center(x, y), (width, height), angle of rotation)
                if rect1[1][0] > rect1[1][1]:
                    area=rect1[1][0]*rect1[1][0]
                else:
                    area=rect1[1][1]*rect1[1][1]                    
                if area>area0:
                     rect=rect1; contour=cont; area0=area            
            
            rot_img = get_digit(thresh, rect)
            
            if save_images : 
                save_image_loc = "ProcessedImages/"+ save_to +"/Train/" + str(int(y[i])) + "_"  + str(int(i)) + image_format
                plt.imsave(save_image_loc, rot_img, cmap="gray")
            x[i] = rot_img
            if ((i+1)%1000==0):
                print((i+1)*100/training_set_size,"% Done")
            i+=1             
    x=x.reshape(training_set_size,sqr_size*sqr_size)
    
    if save_files :
        newpath = 'ProcessedData/' + save_to 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        np.savetxt("ProcessedData/" + save_to + "/train_x.csv",x,fmt='%d',delimiter=',')
        np.savetxt("ProcessedData/" + save_to + "/train_y.csv",y,fmt='%d',delimiter=',')

if with_PCA:
    newpath = 'ProcessedData/' + save_to + '_PCA'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
  
    #Applying Feature scaling and PCA
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x = sc.fit_transform(x)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 100)
    X_train = pca.fit_transform(x)
    np.savetxt("ProcessedData/" + save_to + "_PCA/train_x.csv",X_train,delimiter=',')
    np.savetxt("ProcessedData/" + save_to + "_PCA/train_y.csv",y,fmt='%d',delimiter=',')


# ---------------------------------------
#              TEST DATA
#----------------------------------------
if process_test_data :
    print("Processing the Test Data")
    i=0
    x=np.zeros((test_set_size,sqr_size,sqr_size))
    with open('cs.mcgill.ca/~ksinha4/datasets/kaggle/test_x.csv') as f:
        for line in f: 
            img=np.array(line.split(','))
            img=img.astype(np.float)
            img=img.astype(np.uint8)
            img=np.uint8(img)
            gray=img.reshape(64, 64)
            if save_raw_images:
                save_image_loc = "RawImages/Train/"+ str(int(i)) + "_L"  + str(int(y[i])) + image_format
                plt.imsave(save_image_loc, gray, cmap="gray")
            gray[gray<255] = 0
            thresh = gray
            _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # get contours
        
            area0 = 0
            # for each contour found, draw a rectangle around it on original image
            for cont in contours:           #get the contour and minAreaRectangle with the largest area
                rect1 = cv2.minAreaRect(cont)  # cv2.minAreaRect() returns: (center(x, y), (width, height), angle of rotation)
                if rect1[1][0] > rect1[1][1]:
                    area=rect1[1][0]*rect1[1][0]
                else:
                    area=rect1[1][1]*rect1[1][1]                    
                if area>area0:
                     rect=rect1; contour=cont; area0=area            
            
            rot_img = get_digit(thresh, rect)
            if save_images :  
                save_image_loc = "ProcessedImages/"+ save_to +"/Test/" + str(i) + image_format
                plt.imsave(save_image_loc, rot_img, cmap="gray")
            x[i] = rot_img
            if ((i+1)%1000==0):
                print((i+1)*100/test_set_size,"% Done")
            i+=1            
    x=x.reshape(test_set_size,sqr_size*sqr_size)
    if save_files :
        newpath = 'ProcessedData/' + save_to 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        np.savetxt("ProcessedData/" + save_to + "/test_x.csv",x,fmt='%d',delimiter=',')

if with_PCA:
    newpath = 'ProcessedData/' + save_to + '_PCA'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
  
    #Applying Feature scaling and PCA
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x = sc.fit_transform(x)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 100)
    x_test = pca.fit_transform(x)
    np.savetxt("ProcessedData/" + save_to + "_PCA/test_x.csv",x_test,delimiter=',')
