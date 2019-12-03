import cv2
import os
import glob
import numpy as np

def main():
    for folder in ['train']:
        image_path = 'images_faces_v2/' + folder
        jpg_list = []
        for jpg_list in glob.glob(image_path + '*.jpg'):
            print(jpg_list)

            if jpg_list == "images_faces_v2/train/1.jpg":
                print("[INFO] Passing 1.jpg")
            else:

                img = cv2.imread(jpg_list)
                #print(img.shape)
                height, width, depth = img.shape
                print("[INFO] W=>" + str(width) + "height=>" + str(height))
                bilinear_img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
                height, width, depth = bilinear_img.shape
                print("[INFO] After rescaling W=>" + str(width) + "height=>" + str(height))
                ret = cv2.imwrite(jpg_list, bilinear_img)

                print("Return Value for " + jpg_list + " is " + str(ret))
        

main()