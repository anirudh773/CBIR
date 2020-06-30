import cv2
import numpy as np
import glob
import math
import matplotlib.pyplot as plt
import sys
from skimage import data
from skimage.color import rgb2hsv
# import tensorflow as tf
import io
# from skimage import filters

PHI = 3.1415926
NaN = float('nan')

filter_num = 32
all_theta = np.arange(0, PHI * 2, PHI * 2 / filter_num)

def chkr():
    a, b, c,d = "abcde" ,"xy", 2, 15.06
    print(sys.getsizeof(a))
    print(sys.getsizeof(b))
    print(sys.getsizeof(c))
    print(sys.getsizeof(d))


def display_im(im, name, cmap=''):
    plt.figure()
    if cmap is '':
        plt.imshow(im)
    else:
        plt.imshow(im, cmap=cmap)
    plt.title(name)
    plt.axis('off')

def calc_orientation(im, mask):
    # gabor filter
    kernel_size = (7, 7)
    filtered = np.zeros((hair.shape[0], hair.shape[1], filter_num))
    kernel = [0] * filter_num
    for (idx, theta) in zip(range(0, filter_num), all_theta):
        kernel[idx] = cv2.getGaborKernel(ksize=kernel_size, sigma=1.8, theta=theta, lambd=4,
                                         gamma=1.8 / 2.4)  # , psi=0) # sin
        # print('kernel shape', kernel)
        filtered[:, :, idx] = cv2.filter2D(im, ddepth=-1, kernel=kernel[idx])#ddepth 表示目标图像深度，ddepth=-1 表示生成与原图像深度相同的图像

    # normalize filtered
    denominator = np.sum(filtered * filtered, axis=2, keepdims=True)
    filtered_ = filtered / (np.sqrt(denominator) + 1e-9)

    # calc orientation map
    orientation_init = (filtered_).argmax(axis=2) + 1  # save the theta index, which ranges from 1 to filter_num
    orientation_init[mask == 0] = 0

    # transform to tangent angle
    tangent_angle_init = all_theta[orientation_init - 1] + PHI / 2
    tangent_angle = tangent_angle_init.copy()
    tangent_angle[np.where((tangent_angle >= PHI) & (tangent_angle < PHI * 2))] = tangent_angle[np.where(
        (tangent_angle >= PHI) & (tangent_angle < PHI * 2))] - PHI
    tangent_angle[np.where((tangent_angle >= 2 * PHI))] = tangent_angle[np.where((tangent_angle >= 2 * PHI))] - PHI * 2


    display_im(tangent_angle, 'tangent angle', cmap=plt.cm.jet)

    return tangent_angle

def LBP(image):
    W, H = image.shape
    xx = [-1,  0,  1, 1, 1, 0, -1, -1]
    
    yy = [-1, -1, -1, 0, 1, 1,  1,  0]
    res = np.zeros((W - 2, H - 2),dtype="uint8")
    for i in range(1, W - 2):
        for j in range(1, H - 2):
            temp = ""
            for m in range(8):
                Xtemp = xx[m] + i    
                Ytemp = yy[m] + j
                if image[Xtemp, Ytemp] > image[i, j]:
                    temp = temp + '1'
                else:
                    temp = temp + '0'
            res[i - 1][j - 1] =int(temp, 2)
    display_im(res,'LBP')
    return res

def LBP_circle(image):
    W, H = image.shape
    xx = [0,  1,  2, 1, 0, -1, -2, -1]
    
    yy = [-2, -1, 0, 1, 2, 1,  0,  -1]
    res = np.zeros((W - 3, H - 3),dtype="uint8")
    for i in range(2, W - 3):
        for j in range(2, H - 3):
            temp = ""
            for m in range(8):
                Xtemp = xx[m] + i    
                Ytemp = yy[m] + j
                if image[Xtemp, Ytemp] > image[i, j]:
                    temp = temp + '1'
                else:
                    temp = temp + '0'
            res[i - 1][j - 1] =int(temp, 2)
    display_im(res,'LBP_circle')
    return res

def circular_LBP(src, radius, n_points):
    height = src.shape[0]
    width = src.shape[1]
    dst = src.copy()
    src.astype(dtype=np.float32)
    dst.astype(dtype=np.float32)

    neighbours = np.zeros((1, n_points), dtype=np.uint8)
    lbp_value = np.zeros((1, n_points), dtype=np.uint8)
    for x in range(radius, width - radius - 1):
        for y in range(radius, height - radius - 1):
            lbp = 0.

            for n in range(n_points):
                theta = float(2 * np.pi * n) / n_points
                x_n = x + radius * np.cos(theta)
                y_n = y - radius * np.sin(theta)


                x1 = int(math.floor(x_n))
                y1 = int(math.floor(y_n))

                x2 = int(math.ceil(x_n))
                y2 = int(math.ceil(y_n))


                tx = np.abs(x - x1)
                ty = np.abs(y - y1)
                
                # f(x,y)=f(0,0)(1-x)(1-y)+f(1,0)x(1-y)+f(0,1)(1-x)y+f(1,1)xy

                w1 = (1 - tx) * (1 - ty)
                w2 = tx * (1 - ty)
                w3 = (1 - tx) * ty
                w4 = tx * ty


                neighbour = src[y1, x1] * w1 + src[y2, x1] * w2 + src[y1, x2] * w3 + src[y2, x2] * w4

                neighbours[0, n] = neighbour

            center = src[y, x]

            for n in range(n_points):
                if neighbours[0, n] > center:
                    lbp_value[0, n] = 1
                else:
                    lbp_value[0, n] = 0

            for n in range(n_points):
                lbp += lbp_value[0, n] * 2**n


            dst[y, x] = int(lbp / (2**n_points-1) * 255)
    display_im(dst,'LBP_circle_'+'R='+str(radius)+'_P='+str(n_points))
    return dst

def value_rotation(num):
    value_list = np.zeros((8), np.uint8)
    temp = int(num)
    value_list[0] = temp
    for i in range(7):
        temp = ((temp << 1) | int(temp / 128)) % 256
        value_list[i+1] = temp
    return np.min(value_list)

def rotation_invariant_LBP(src):
    height = src.shape[0]
    width = src.shape[1]
    dst = src.copy()

    lbp_value = np.zeros((1, 8), dtype=np.uint8)
    neighbours = np.zeros((1, 8), dtype=np.uint8)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            neighbours[0, 0] = src[y - 1, x - 1]
            neighbours[0, 1] = src[y - 1, x]
            neighbours[0, 2] = src[y - 1, x + 1]
            neighbours[0, 3] = src[y, x - 1]
            neighbours[0, 4] = src[y, x + 1]
            neighbours[0, 5] = src[y + 1, x - 1]
            neighbours[0, 6] = src[y + 1, x]
            neighbours[0, 7] = src[y + 1, x + 1]

            center = src[y, x]

            for i in range(8):
                if neighbours[0, i] > center:
                    lbp_value[0, i] = 1
                else:
                    lbp_value[0, i] = 0

            lbp = lbp_value[0, 0] * 1 + lbp_value[0, 1] * 2 + lbp_value[0, 2] * 4 + lbp_value[0, 3] * 8 \
                  + lbp_value[0, 4] * 16 + lbp_value[0, 5] * 32 + lbp_value[0, 6] * 64 + lbp_value[0, 0] * 128


            dst[y, x] = value_rotation(lbp)
    display_im(dst,'LBP_invariant')
    return dst

def sharpening(image):
    W, H = image.shape
    xx = [-1,  0,  1, 1, 1, 0, -1, -1, 0]
    
    yy = [-1, -1, -1, 0, 1, 1,  1,  0, 0]
    
    w = [-1,  -1,  -1, -1, -1, -1, -1, -1, 9]
    
    res = np.zeros((W - 2, H - 2),dtype="uint8")
    for i in range(1, W - 2):
        for j in range(1, H - 2):
            temp = 0
            for m in range(9):
                Xtemp = xx[m] + i    
                Ytemp = yy[m] + j
                temp=temp+image[Xtemp, Ytemp]*w[m]
            res[i - 1][j - 1] =temp
    return res



def getColorHist(img):
	r=img[:,:,0].flatten()
	g=img[:,:,1].flatten()
	b=img[:,:,2].flatten()
	plt.figure(num='colorhist', figsize=(21,8))
	plt.subplot(1,3,1)
	plt.title('red')
	n, bins, patches = plt.hist(r, bins=256, edgecolor='None',facecolor='red')
	plt.subplot(132)
	plt.title('green')
	n, bins, patches = plt.hist(g, bins=256, edgecolor='None',facecolor='green')
	plt.subplot(133)
	plt.title('blue')
	n, bins, patches = plt.hist(b, bins=256, edgecolor='None',facecolor='blue')  
	plt.show()

def autoCorrelogram(img,d,max_color):
	w=img.shape[0]
	h=img.shape[1]
	autoCorrelogram=np.zeros([w,h]) 
	temp=np.ones([w,h])
	ext_img=np.zeros([w+2*d,h+2*d])
	ext_img[d:d+w,d:d+h]=img 
	
	for i in range(2*d+1):
		if i==0 or i==2*d:
			for j in range(2*d+1):
				autoCorrelogram=autoCorrelogram+temp*(img==ext_img[i:i+w,j:j+h])
		else:
			autoCorrelogram=autoCorrelogram+temp*(img==ext_img[i:i+w,:h])
			autoCorrelogram=autoCorrelogram+temp*(img==ext_img[i:i+w,2*d:2*d+h])
	
	autoHist=[]
	lastSum=0
	for i in range(1,max_color+1):
		autoHist.append((autoCorrelogram * (img<i*256/max_color)).sum()-lastSum)
		lastSum = (autoCorrelogram * (img<i*256/max_color)).sum()
	return autoHist

def rgbAtuo(img,d,max_color):
	r=img[:,:,0]
	g=img[:,:,1]
	b=img[:,:,2]
	rf=[]
	gf=[]
	bf=[]
	rf=autoCorrelogram(r,d,max_color)
	gf=autoCorrelogram(r,d,max_color)
	bf=autoCorrelogram(r,d,max_color)
	return rf, gf, bf


if __name__ == '__main__':
    
    #rgb to hsv
    im = cv2.imread('images/lenna.jpg')
    hsv_img = rgb2hsv(im)
    hue_img = hsv_img[:, :, 0]
    sat_img = hsv_img[:, :, 1]
    value_img = hsv_img[:, :, 2]

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(8, 2))

    ax0.imshow(im)
    ax0.set_title("RGB image")
    ax0.axis('off')
    ax1.imshow(hue_img, cmap='hsv')
    ax1.set_title("Hue channel")
    ax1.axis('off')
    ax2.imshow(value_img)
    ax2.set_title("Value channel")
    ax2.axis('off')
    ax3.imshow(sat_img)
    ax3.set_title("Saturation channel")
    ax3.axis('off')
    fig.tight_layout()
    plt.show()

    # auto correlogram
    rf, gf, bf = rgbAtuo(im, 1, 64)
    plt.figure(num='colorhist', figsize=(21,8))
    plt.subplot(1,3,1)
    plt.title('red')
    n, bins, patches = plt.hist(rf, bins=256, edgecolor='None',facecolor='red')
    plt.subplot(132)
    plt.title('green')
    n, bins, patches = plt.hist(gf, bins=256, edgecolor='None',facecolor='green')
    plt.subplot(133)
    plt.title('blue')
    n, bins, patches = plt.hist(bf, bins=256, edgecolor='None',facecolor='blue')  
    plt.show()


    # LSB
    mask = cv2.imread('images/lenna.jpg')
    
    b,g,r = cv2.split(mask)
    mask = cv2.merge((r,g,b))
    b,g,r = cv2.split(im)
    im = cv2.merge((r,g,b))
    display_im(im, 'im')
    display_im(cv2.merge((sharpening(r),sharpening(g),sharpening(b))),'sharpening')
    
    #get hair part
    mask[mask > 50] = 255
    mask[mask <= 50] = 0
    
    # get hair image
    mask_pts = np.where(mask == 255)
    pt_row, pt_col = np.array(mask_pts[0]), np.array(mask_pts[1])
    mask_pts = np.concatenate((pt_col[:, np.newaxis], pt_row[:, np.newaxis]), axis=1)
    x, y, w, h = cv2.boundingRect(np.array(mask_pts, dtype='float32'))
    hair = im[y : y + h, x : x + w, :]
    hair[np.where(mask[y : y + h, x : x + w, :] != 255)] = 0

    # plt.imshow(hair.astype(np.uint8))
    display_im(hair.astype(np.uint8), 'hair')
    # gabor filter, get orientation and confidence
    hair_gray = cv2.cvtColor(hair, cv2.COLOR_RGB2GRAY)
    orientation = calc_orientation(hair_gray, mask[y : y + h, x : x + w, 0])#gabor filter
    orientation2 = LBP(hair_gray)#LBP
    orientation3 = LBP_circle(hair_gray)#circle LBP without interpolation
    rotation_invariant_LBP(src=hair_gray)#invariant LBP 
    circular_LBP(hair_gray, radius=3, n_points=32)#circle LBP with interpolation
    plt.show()
    while(True):
        pass