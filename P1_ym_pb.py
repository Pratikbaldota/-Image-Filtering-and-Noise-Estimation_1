#!/bin/env python3
import cv2
import os
import glob
class project1():
    def __init__(self,source_img_path):
        self.src = source_img_path
        os.chdir(source_img_path)
        self.gray_path = source_img_path + "/Gray_scale/gray_"
        self.temp_der_path = source_img_path + "/Gray_scale/temp_der/der_new_"
        self.img_names = []
        for f in glob.glob("*.jpg"):
            self.img_names.append(f)
        self.img_names = sorted(self.img_names)
    
    def read_gray_scale(self):
        for i in range(len(self.img_names)):
            rgb = cv2.imread(self.img_names[i],cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(self.gray_path + str(i)+ '.jpg',rgb)
    
    def first_derivate(self):
        for i in range(1,len(self.img_names)):
            gray_prev = cv2.imread(self.gray_path + str(abs(1 - i)) + '.jpg')
            gray_curr = cv2.imread(self.gray_path + str(i)+ '.jpg')
            temp_derivate = cv2.subtract(gray_prev,gray_curr)
            cv2.imwrite(self.temp_der_path + str(i)+ '.jpg',temp_derivate)   
        print(temp_derivate)
            # cv2.imshow("temp_derivate",temp_derivate)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # temp_derivate = gray_prev - gray_curr
        # for i in range(1,len(self.img_names)):
        #     gray = cv2.imread(self.gray_path + str(i)+ '.jpg')
        #     print("prev",abs(1-i),end=" ")
        #     print("curr",i)
        #     temp_derivate = cv2.subtract(gray[abs(1-i)],gray[i])
        #     cv2.imwrite(self.temp_der_path + str(i)+ '.jpg',temp_derivate)   



if __name__ == "__main__":
    office_imgs = "/home/yash/Documents/Computer_VIsion/Project 1/Office/Office"
    red_chair_imgs = "/home/yash/Documents/Computer_VIsion/Project 1/RedChair/RedChair"
    motion_detect = project1(office_imgs)
    # motion_detect.read_gray_scale()
    motion_detect.first_derivate()