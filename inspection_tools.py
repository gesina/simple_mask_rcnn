#! /bin/python3

# Some inspection tools

import mnist_mask
import cv2
import os

# check image formats
def check_image_formats(folder):
    for path, dirs, files in os.walk(folder):
        for filename in files:
            #print(path, filename)
            filepath = os.path.join(path, filename)
            with open(filepath, 'rb') as f:
                check_chars = f.read()[-2:]
            if check_chars != b'\xff\xd9':
                print('Not complete image: ', filepath)
            else:
                imrgb = cv2.imread(filepath, 1)

def full_check(*folders):
    print("Doing a check whether used images are proper .jpg files:")
    for folder in folders:
        print("Checking images in", folder, "...")
        check_image_formats(folder)
            

def inspect_data(num_images=40, test_folder="test"):
    data = mnist_mask.load_labeled_data()
    for i in range(0, min(num_images, len(data[0]))):
        img = data[0][i]
        matches = data[1][i]
        img = mnist_mask.draw_masks_and_labels(img, matches)
        if img is None:
            print(i, "empty")
            exit(1)
        newimgfile = os.path.join(test_folder, "test"+str(i)+".jpg")
        print("Writing", newimgfile)
        cv2.imwrite(newimgfile, img)

        
if __name__ == "__main__":
    # check image data format
    #full_check("data/textures", "data/grids", "data/stains")
    #full_check("data/mask_rcnn/images")
    inspect_data(num_images=200)
