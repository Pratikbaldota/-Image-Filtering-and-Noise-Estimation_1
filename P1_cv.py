#!/bin/env python3
import cv2
import os
import glob
from matplotlib import pyplot as plt

class project1():
    def __init__(self, source_img_path):
        self.src = source_img_path
        os.chdir(source_img_path)
        self.gray_path = source_img_path + "/Gray_scale/gray_"
        self.temp_der_path = source_img_path + "/Gray_scale/temp_der/der_new_"
        self.temp_der_path3 = source_img_path + "/Gray_scale/temp_der/filt3/der_new3_"
        self.temp_der_path5 = source_img_path + "/Gray_scale/temp_der/filt5/der_new5_"
        self.temp_der_path6 = source_img_path + "/Gray_scale/temp_der/filt_gauss/der_new6_"
        # /home/yash/Documents/Computer_VIsion/Project 1/CV_Project1/Office/Office/Gray_scale/temp_der/filt_gauss
        self.img_names = []
        for f in glob.glob("*.jpg"):
            self.img_names.append(f)
        self.img_names = sorted(self.img_names)
    
    def read_gray_scale(self):
        for i in range(len(self.img_names)):
            rgb = cv2.imread(self.img_names[i], cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(self.gray_path + str(i) + '.jpg', rgb)
    
    def first_derivative(self):
        for i in range(1, len(self.img_names)):
            gray_prev = cv2.imread(self.gray_path + str(abs(1 - i)) + '.jpg')
            gray_curr = cv2.imread(self.gray_path + str(i) + '.jpg')
            temp_derivative = cv2.subtract(gray_prev, gray_curr)
            ret,thresh1 = cv2.threshold(temp_derivative,40,255,cv2.THRESH_BINARY)
            cv2.imwrite(self.temp_der_path + str(i) + '.jpg', thresh1)
        
    def threethree_derivative(self):
        for i in range(1, len(self.img_names)):
            gray_prev = cv2.imread(self.gray_path + str(abs(1 - i)) + '.jpg')
            gray_curr = cv2.imread(self.gray_path + str(i) + '.jpg')
            
            # Apply spatial smoothing filter
            smooth_curr = cv2.blur(gray_curr, (3, 3))
            smooth_prev = cv2.blur(gray_prev, (3, 3))
            
            temp_derivative = cv2.subtract(smooth_prev, smooth_curr)
            # cv2.imwrite(self.temp_der_path3 + str(i) + '.jpg', temp_derivative)
            ret,thresh1 = cv2.threshold(temp_derivative,40,255,cv2.THRESH_BINARY)
            cv2.imwrite(self.temp_der_path3 + str(i) + '.jpg', thresh1)
    
    def fivefive_derivative(self):
        for i in range(1, len(self.img_names)):
            gray_prev = cv2.imread(self.gray_path + str(abs(1 - i)) + '.jpg')
            gray_curr = cv2.imread(self.gray_path + str(i) + '.jpg')
            
            # Apply 5x5 spatial smoothing filter
            smooth_curr = cv2.blur(gray_curr, (5, 5))
            smooth_prev = cv2.blur(gray_prev, (5, 5))
            
            temp_derivative = cv2.subtract(smooth_prev, smooth_curr)
            # cv2.imwrite(self.temp_der_path5 + str(i) + '.jpg', temp_derivative)
            ret,thresh1 = cv2.threshold(temp_derivative,40,255,cv2.THRESH_BINARY)
            cv2.imwrite(self.temp_der_path5 + str(i) + '.jpg', thresh1)
    
    def apply_gaussian_filter(self, sigma):
        for i in range(1, len(self.img_names)):
            gray_prev = cv2.imread(self.gray_path + str(abs(1 - i)) + '.jpg')
            gray_curr = cv2.imread(self.gray_path + str(i) + '.jpg')

            kernel = cv2.getGaussianKernel(3 * sigma, sigma)

            guass_curr = cv2.filter2D(gray_curr, -1, kernel * kernel.T)
            guass_prev = cv2.filter2D(gray_prev, -1, kernel * kernel.T)

            temp_derivative = cv2.subtract(guass_prev, guass_curr)
            ret,thresh1 = cv2.threshold(temp_derivative,40,255,cv2.THRESH_BINARY)
            cv2.imwrite(self.temp_der_path6 + str(i) + '.jpg', thresh1)
            # cv2.imwrite(self.temp_der_path6 + str(i) + '.jpg', temp_derivative)

    def analysis_filts(self):
        # print(self.temp_der_path + str(141) + '.jpg')
        img_no_filt = cv2.imread(self.temp_der_path + str(141) + '.jpg',cv2.IMREAD_GRAYSCALE)
        no_filt = cv2.calcHist([img_no_filt],[0], None, [256], [0,256])
        img_filt_3 = cv2.imread(self.temp_der_path3 + str(141) + '.jpg',cv2.IMREAD_GRAYSCALE)
        filt_3 = cv2.calcHist([img_filt_3],[0], None, [256], [0,256])
        img_filt_5 = cv2.imread(self.temp_der_path5 + str(141) + '.jpg',cv2.IMREAD_GRAYSCALE)
        filt_5 = cv2.calcHist([img_filt_5],[0], None, [256], [0,256])
        img_filt_guass = cv2.imread(self.temp_der_path6 + str(141) + '.jpg',cv2.IMREAD_GRAYSCALE)
        filt_6 = cv2.calcHist([img_filt_guass],[0], None, [256], [0,256])
        plt.subplot()
        # plt.figure()
        # plt.axis("off")
        plt.plot(no_filt)
        plt.plot(filt_3)
        plt.plot(filt_5)
        plt.plot(filt_6)
        plt.legend()
        plt.show()
    

if __name__ == "__main__":

    office_imgs = "/home/yash/Documents/Computer_VIsion/Project 1/CV_Project1/Office/Office"
    red_chair_imgs = "/home/yash/Documents/Computer_VIsion/Project 1/CV_Project1/RedChair/RedChair"
    motion_detect = project1(office_imgs)
    # motion_detect.read_gray_scale()
    motion_detect.first_derivative()
    motion_detect.threethree_derivative()
    motion_detect.fivefive_derivative()
    motion_detect.apply_gaussian_filter(0)
    motion_detect.analysis_filts()