#! /bin/python3

# Some inspection tools

import mnist_mask
import cv2
import os
from config import Config
import numpy as np


# check image formats
def check_image_formats(folder):
    for path, dirs, files in os.walk(folder):
        for filename in files:
            # print(path, filename)
            filepath = os.path.join(path, filename)
            with open(filepath, 'rb') as f:
                check_chars = f.read()[-2:]
            if check_chars != b'\xff\xd9':
                print('Not complete image: ', filepath)


def full_check(*folders):
    print("Doing a check whether used images are proper .jpg files:")
    for folder in folders:
        print("Checking images in", folder, "...")
        check_image_formats(folder)


def inspect_data(num_images=40, test_folder="test"):
    images, all_matches, _ = mnist_mask.load_labeled_data()
    for i in range(0, min(num_images, len(images))):
        img = images[i]
        matches = all_matches[i]
        img = mnist_mask.draw_masks_and_labels(img, matches)
        if img is None:
            print(i, "empty")
            exit(1)
        newimgfile = os.path.join(test_folder, "test" + str(i) + ".jpg")
        print("Writing", newimgfile)
        cv2.imwrite(newimgfile, img)


def inspect_center_points(pt_width=None):
    default_config = Config()
    if pt_width is None:
        pt_width = int(default_config.DOWNSCALING_FACTOR / 2 - 1)
    # empty image
    img_width = default_config.IMAGE_SHAPE[1]
    img_height = default_config.IMAGE_SHAPE[0]
    img_channels = 1
    image = np.zeros(shape=(img_height, img_width, img_channels))

    # center points
    center_points = default_config.CENTER_POINTS.tolist()
    center_points = list(map(lambda p: (int(round(p[0] * img_width)),
                                        int(round(p[1] * img_height))),
                             center_points))
    for pt in center_points:
        x, y = pt
        image[y - pt_width:y + pt_width, x - pt_width:x + pt_width, :] = 255
    cv2.imwrite("test.png", image)


def inspect_anchors():
    default_config = Config()

    # empty image
    img_width = default_config.IMAGE_SHAPE[1]
    img_height = default_config.IMAGE_SHAPE[0]
    img_channels = 3

    # anchors
    anchors = default_config.ANCHOR_BOXES
    anchors = list(map(lambda box: ((int(round(box[0] * img_width)),
                                     int(round(box[1] * img_height))),
                                    (int(round(box[2] * img_width)),
                                     int(round(box[3] * img_height)))),
                       anchors))

    # test dir
    if not os.path.isdir("temp"):
        os.mkdir("temp")
    for i in range(0, len(anchors)):
        image = np.zeros(shape=(img_height, img_width, img_channels))
        a = anchors[i]
        cv2.rectangle(image, a[0], a[1], color=(255, 0, 0))
        cv2.imwrite("temp/test{:04d}.png".format(i), image)


if __name__ == "__main__":
    # check image data format
    # full_check("data/textures", "data/grids", "data/stains")
    # full_check("data/mask_rcnn/images")
    # inspect_data(num_images=200)
    inspect_center_points(pt_width=1)
    inspect_anchors()
    config = Config()
    print(config.NUM_ANCHORS)