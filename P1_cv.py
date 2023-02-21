#!/bin/env python3
import cv2
import os
import glob
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage

class project1():
    def __init__(self, source_img_path):
        self.src = source_img_path
        os.chdir(source_img_path)
        self.gray_path = source_img_path + "/Gray_scale/gray_"
        self.temp_der_path = source_img_path + "/Gray_scale/temp_der/der_new_"
        self.temp_der_path3 = source_img_path + "/Gray_scale/temp_der/filt3/der_new3_"
        self.temp_der_path5 = source_img_path + "/Gray_scale/temp_der/filt5/der_new5_"
        self.temp_der_path6 = source_img_path + "/Gray_scale/temp_der/filt_gauss/der_new6_"
        self.derived_frames = source_img_path + "/Gray_scale/derived_frames/der_"
        self.derived_guass = source_img_path + "/Gray_scale/derived_guass/der_"
        self.derived_masks = source_img_path + "/Gray_scale/derived_mask/der_maks_"
        # /home/yash/Documents/Computer_VIsion/Project 1/CV_Project1/Office/Office/Gray_scale/temp_der/filt_gauss
        self.img_names = []
        for f in glob.glob("*.jpg"):
            self.img_names.append(f)
        self.img_names = sorted(self.img_names)
    
    def read_gray_scale(self):
        for i in range(len(self.img_names)):
            rgb = cv2.imread(self.img_names[i], cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(self.gray_path + str(i) + '.jpg', rgb)
    
    def temp_subtract(self):
        for i in range(1, len(self.img_names)):
            gray_prev = cv2.imread(self.gray_path + str(abs(1 - i)) + '.jpg')
            gray_curr = cv2.imread(self.gray_path + str(i) + '.jpg')
            temp_derivative = cv2.subtract(gray_prev, gray_curr)
            # ret,thresh1 = cv2.threshold(temp_derivative,40,255,cv2.THRESH_BINARY)
            # cv2.imwrite(self.temp_der_path + str(i) + '.jpg', thresh1)
            cv2.imwrite(self.temp_der_path + str(i) + '.jpg', temp_derivative)
        
    def threethree_derivative(self):
        for i in range(len(self.img_names)):
            # gray_prev = cv2.imread(self.gray_path + str(abs(1 - i)) + '.jpg')
            gray_curr = cv2.imread(self.gray_path + str(i) + '.jpg')
            
            # Apply spatial smoothing filter
            smooth_curr = cv2.blur(gray_curr, (3, 3))
            # smooth_prev = cv2.blur(gray_prev, (3, 3))
            
            # temp_derivative = cv2.subtract(smooth_prev, smooth_curr)
            # cv2.imwrite(self.temp_der_path3 + str(i) + '.jpg', temp_derivative)
            # ret,thresh1 = cv2.threshold(temp_derivative,40,255,cv2.THRESH_BINARY)
            cv2.imwrite(self.temp_der_path3 + str(i) + '.jpg', smooth_curr)
    
    def fivefive_derivative(self):
        for i in range(len(self.img_names)):
            # gray_prev = cv2.imread(self.gray_path + str(abs(1 - i)) + '.jpg')
            gray_curr = cv2.imread(self.gray_path + str(i) + '.jpg')
            
            # Apply 5x5 spatial smoothing filter
            smooth_curr = cv2.blur(gray_curr, (5, 5))
            # smooth_prev = cv2.blur(gray_prev, (5, 5))
            
            # temp_derivative = cv2.subtract(smooth_prev, smooth_curr)
            # cv2.imwrite(self.temp_der_path5 + str(i) + '.jpg', temp_derivative)
            # ret,thresh1 = cv2.threshold(temp_derivative,40,255,cv2.THRESH_BINARY)
            cv2.imwrite(self.temp_der_path5 + str(i) + '.jpg', smooth_curr)
    
    def apply_gaussian_filter(self, sigma):
        for i in range(len(self.img_names)):
            # gray_prev = cv2.imread(self.gray_path + str(abs(1 - i)) + '.jpg')
            gray_curr = cv2.imread(self.gray_path + str(i) + '.jpg')

            kernel = cv2.getGaussianKernel(3 * sigma, sigma)

            guass_curr = cv2.filter2D(gray_curr, -1, kernel * kernel.T)
            # guass_prev = cv2.filter2D(gray_prev, -1, kernel * kernel.T)

            # temp_derivative = cv2.subtract(guass_prev, guass_curr)
            # ret,thresh1 = cv2.threshold(temp_derivative,40,255,cv2.THRESH_BINARY)
            cv2.imwrite(self.temp_der_path6 + str(i) + '.jpg', guass_curr)
            # cv2.imwrite(self.temp_der_path6 + str(i) + '.jpg', temp_derivative)

    def EST_NOISE(self):
        list_sig = []
        for i in range(len(self.img_names)):
            print(i)
            images = cv2.imread(self.derived_frames + str(i) + '.jpg',0).astype(np.float64)
            num = images.shape[0]
            m_e_bar = sum(images)/num
            m_sigma = np.sqrt(sum((images - m_e_bar)**2) / (num - 1))
            list_sig.append(np.std(m_sigma))
        return np.median(list_sig)
        # return m_sigma
    
    def temp_der(self):
        # gray = cv2.imread(self.temp_der_path3 + str(141) + '.jpg',0).astype(np.float64)
        # Kx = [-0.5,0.0,0.5]
        # Fx = ndimage.convolve1d(gray, weights=Kx,axis=1,mode='constant')
        # cv2.imwrite(self.derived_frames + str(141) + '.jpg',Fx)

        # gray = cv2.imread(self.temp_der_path5 + str(142) + '.jpg',0).astype(np.float64)
        # Kx = [-0.5,0.0,0.5]
        # Fx = ndimage.convolve1d(gray, weights=Kx,axis=1,mode='constant')
        # cv2.imwrite(self.derived_frames + str(142) + '.jpg',Fx)

        # gray = cv2.imread(self.temp_der_path6 + str(143) + '.jpg',0).astype(np.float64)
        # Kx = [-0.5,0.0,0.5]
        # Fx = ndimage.convolve1d(gray, weights=Kx,axis=1,mode='constant')
        # cv2.imwrite(self.derived_frames + str(143) + '.jpg',Fx)


        for i in range(len(self.img_names)):
            gray = cv2.imread(self.gray_path + str(i) + '.jpg',0).astype(np.float64)
            Kx = [-0.5,0.0,0.5]
            Fx = ndimage.convolve1d(gray, weights=Kx,axis=1,mode='constant')
            cv2.imwrite(self.derived_frames + str(i) + '.jpg',Fx)
    
    def temp_der_guass(self,sigma):
        z=np.ceil(3*sigma).astype(int)
        k=np.arange(-z,z+1)
        gauss = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-1*(k)**2/(2*sigma**2))
        gauss /= np.sum(gauss)
        gauss_d = -(k/sigma**2)*gauss
        for i in range(len(self.img_names)):
            gray1 = cv2.imread(self.gray_path + str(i) + '.jpg',0).astype(np.float64)
            gray2 = cv2.imread(self.gray_path + str(i) + '.jpg',0).astype(np.float64)
            Fx1 = ndimage.convolve1d(gray1, weights=gauss_d,axis=1,mode='constant')
            Fx2 = ndimage.convolve1d(gray2, weights=gauss_d,axis=1,mode='constant')
            diff = np.abs(Fx2 - Fx1)
            threshold = self.EST_NOISE()
            mask = (diff > threshold).astype(np.uint8) * 255
            # display the mask overlaid on the original frame
            result_1 = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
            result_1[mask != 0] = (0, 0, 255)
            cv2.imwrite(self.derived_guass + str(i) + '.jpg',result_1)

    def thres_mask(self,k):
        threshold = self.EST_NOISE()
        
        temp = cv2.imread(self.derived_frames + str(141) + '.jpg',0).astype(np.float64)
        mask = np.zeros(temp.shape,dtype=np.uint8)
        mask[(np.abs(temp) >= 0.5*threshold)] = 1
        cv2.imwrite(self.derived_masks + str(141) + '.jpg',mask*255)

        temp = cv2.imread(self.derived_frames + str(140) + '.jpg',0).astype(np.float64)
        mask = np.zeros(temp.shape,dtype=np.uint8)
        mask[(np.abs(temp) > 3*threshold)] = 1
        cv2.imwrite(self.derived_masks + str(140) + '.jpg',mask*255)

        temp = cv2.imread(self.derived_frames + str(142) + '.jpg',0).astype(np.float64)
        mask = np.zeros(temp.shape,dtype=np.uint8)
        mask[(np.abs(temp) >= 0.5*threshold)] = 1
        cv2.imwrite(self.derived_masks + str(142) + '.jpg',mask*255)

        temp = cv2.imread(self.derived_frames + str(143) + '.jpg',0).astype(np.float64)
        mask = np.zeros(temp.shape,dtype=np.uint8)
        mask[(np.abs(temp) >= 0.5*threshold)] = 1
        cv2.imwrite(self.derived_masks + str(143) + '.jpg',mask*255)


        # for i in range(len(self.img_names)):
            # temp = cv2.imread(self.derived_frames + str(i) + '.jpg',0).astype(np.float64)
            # mask = np.zeros(temp.shape,dtype=np.uint8)
            # mask[(np.abs(temp) > threshold)] = 1
            # cv2.imwrite(self.derived_masks + str(i) + '.jpg',mask*255)


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
        plt.plot(no_filt)
        plt.plot(filt_3)
        plt.plot(filt_5)
        plt.plot(filt_6)
        plt.legend()
        plt.show()
    
    def combine_mask(self):
        mask = cv2.imread(self.derived_masks + str(140) + '.jpg')
        frame = cv2.imread(self.gray_path + str(140) + '.jpg')
        motion_highlight = np.multiply(frame,mask)
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(frame)
        axs[0].set_title('Original Frame')
        axs[1].imshow(motion_highlight, cmap='jet')
        axs[1].set_title('Motion Highlight')
        axs[2].imshow(mask)
        axs[2].set_title('Binary Mask')
        plt.show()
    
    def results(self):
        # read in the current frame and the previous frame
        frame1 = cv2.imread(self.gray_path + str(140) + '.jpg',0)
        frame2 = cv2.imread(self.gray_path + str(141) + '.jpg',0)

        # apply the 0.5*[-1,0,1] filter to each frame
        Kx = np.array([-0.5, 0.0, 0.5])
        Fx1 = ndimage.convolve1d(frame1.astype(np.float64), Kx, axis=1, mode='constant')
        Fx2 = ndimage.convolve1d(frame2.astype(np.float64), Kx, axis=1, mode='constant')

        # compute the absolute difference between the two filtered frames
        diff = np.abs(Fx2 - Fx1)

        # threshold the absolute difference to create a binary mask
        threshold = 10
        mask = (diff > threshold).astype(np.uint8) * 255

        # display the mask overlaid on the original frame
        result_1 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)
        result_1[mask != 0] = (0, 0, 255)
        cv2.imwrite("detection.jpg",result_1)
        detect = cv2.imread("detection.jpg")
        plt.imshow('Result', detect)
        plt.show()
        # cv2.waitKey(0)


if __name__ == "__main__":

    office_imgs = "/home/yash/Documents/Computer_VIsion/Project 1/CV_Project1/Office/Office"
    red_chair_imgs = "/home/yash/Documents/Computer_VIsion/Project 1/CV_Project1/RedChair/RedChair"
    motion_detect = project1(office_imgs)
    # motion_detect.temp_subtract()
    # motion_detect.temp_der()
    # threshold = motion_detect.EST_NOISE()
    # motion_detect.thres_mask(3)

    # original_frame = "/home/yash/Documents/Computer_VIsion/Project 1/CV_Project1/Office/Office/Gray_scale/derived_mask/der_maks_140.jpg"
    # subtracted_img = "/home/yash/Documents/Computer_VIsion/Project 1/CV_Project1/Office/Office/Gray_scale/derived_mask/der_maks_141.jpg"
    # derived_img = "/home/yash/Documents/Computer_VIsion/Project 1/CV_Project1/Office/Office/Gray_scale/derived_mask/der_maks_142.jpg"
    # derived_guass = "/home/yash/Documents/Computer_VIsion/Project 1/CV_Project1/Office/Office/Gray_scale/derived_mask/der_maks_143.jpg"
    # o = cv2.imread(original_frame)
    # s = cv2.imread(subtracted_img)
    # f = cv2.imread(derived_img)
    # g = cv2.imread(derived_guass)
    # # cv2.imshow("output",[])
    # fig,axs = plt.subplots(2,2)
    # axs[0, 0].imshow(o)
    # axs[0, 0].set_title('5 times est_noise')
    # axs[0, 1].imshow(s)
    # axs[0, 1].set_title('3 times est_noise')
    # axs[1, 0].imshow(f)
    # axs[1, 0].set_title('10 times est_noise')
    # axs[1, 1].imshow(g)
    # axs[1, 1].set_title('20 times est_noise')
    # plt.show()
    # motion_detect.temp_der_guass(1)

    # original_frame = "/home/yash/Documents/Computer_VIsion/Project 1/CV_Project1/Office/Office/Gray_scale/gray_141.jpg"
    # subtracted_img = "/home/yash/Documents/Computer_VIsion/Project 1/CV_Project1/Office/Office/Gray_scale/temp_der/der_new_141.jpg"
    # derived_img = "/home/yash/Documents/Computer_VIsion/Project 1/CV_Project1/Office/Office/Gray_scale/derived_frames/der_141.jpg"
    # derived_guass = "/home/yash/Documents/Computer_VIsion/Project 1/CV_Project1/Office/Office/Gray_scale/derived_guass/der_141.jpg"
    # o = cv2.imread(original_frame)
    # s = cv2.imread(subtracted_img)
    # f = cv2.imread(derived_img)
    # g = cv2.imread(derived_guass)
    # # cv2.imshow("output",[])
    # fig,axs = plt.subplots(2,2)
    # axs[0, 0].imshow(o)
    # axs[0, 0].set_title('original_img')
    # axs[0, 1].imshow(s)
    # axs[0, 1].set_title('subtracted_img')
    # axs[1, 0].imshow(f)
    # axs[1, 0].set_title('1D derivative')
    # axs[1, 1].imshow(g)
    # axs[1, 1].set_title('1D gaussian derivative')
    # plt.show()
    # ax1 = fig.add_subplot(2,2)
    # ax1.imshow(original_frame)
    # ax1.axis('off')
    # ax1.set_title("Original", fontsize=20)
    
    # ax2 = fig.add_subplot(132)
    # ax2.imshow(subtracted_img, cmap=plt.cm.gray)
    # ax2.axis('off')
    # ax2.set_title("subtracted", fontsize=20)
    
    # ax3 = fig.add_subplot(133)
    # ax3.imshow(derived_img)
    # ax3.axis('off')
    # ax3.set_title("1D derivative", fontsize=20)
    
    # ax3 = fig.add_subplot(134)
    # ax3.imshow(derived_guass)
    # ax3.axis('off')
    # ax3.set_title("1D derivative of guass", fontsize=20)

    # plt.show()
    # motion_detect.results()
    motion_detect.temp_der_guass(2.5)
    # motion_detect.thres_mask(1)
    # motion_detect.combine_mask()

    # motion_detect.read_gray_scale()
    # motion_detect.first_derivative()
    # motion_detect.threethree_derivative()
    # motion_detect.fivefive_derivative()
    # motion_detect.apply_gaussian_filter(3)
    # motion_detect.temp_der()
    # original_frame = "/home/yash/Documents/Computer_VIsion/Project 1/CV_Project1/Office/Office/Gray_scale/derived_mask/der_maks_140.jpg"
    # subtracted_img = "/home/yash/Documents/Computer_VIsion/Project 1/CV_Project1/Office/Office/Gray_scale/derived_mask/der_maks_141.jpg"
    # derived_img = "/home/yash/Documents/Computer_VIsion/Project 1/CV_Project1/Office/Office/Gray_scale/derived_mask/der_maks_142.jpg"
    # derived_guass = "/home/yash/Documents/Computer_VIsion/Project 1/CV_Project1/Office/Office/Gray_scale/derived_mask/der_maks_143.jpg"
    # o = cv2.imread(original_frame)
    # s = cv2.imread(subtracted_img)
    # f = cv2.imread(derived_img)
    # g = cv2.imread(derived_guass)
    # # cv2.imshow("output",[])
    # fig,axs = plt.subplots(2,2)
    # axs[0, 0].imshow(o)
    # axs[0, 0].set_title('No smoothing derivative')
    # axs[0, 1].imshow(s)
    # axs[0, 1].set_title('3x3 smoothing derivative')
    # axs[1, 0].imshow(f)
    # axs[1, 0].set_title('5x5 smoothing derivative')
    # axs[1, 1].imshow(g)
    # axs[1, 1].set_title('Guassian Smoothing derivative')
    # plt.show()