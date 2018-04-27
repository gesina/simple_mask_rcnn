#! /bin/python3

from model import MaskRCNNWrapper, Config

import cv2




def to_rpn_cls_and_rpn_reg(matches, image_shape):
    rpn_cls = []
    rpn_reg = []

    for label, ((abs_x1, abs_y1), (abs_x2, abs_y2)), mask in matches:
        next_anchor = best_anchor()
        x1, y1 = to_abs_coordinates(abs_x1, abs_y1, image_shape)
        x2, y2 = to_abs_coordinates(abs_x2, abs_y2, image_shape)

    return rpn_cls, rpn_reg


if __name__ == "__main__":
    config = Config()
    wrapper = MaskRCNNWrapper(config)
    wrapper.compile_model()

    images = images.map(lambda img: cv2.reshape(img, config.IMAGE_SHAPE))
    rpn_cls, rpn_reg = to_rpn_cls_and_rpn_reg(matches, config.IMAGE_SHAPE)
