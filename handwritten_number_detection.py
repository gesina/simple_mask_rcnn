#! /bin/python3

import h5py
from model import MaskRCNNWrapper
from config import Config
import utils

MODEL_FILE = "rpn.h5"
MODEL_WEIGHTS_FILE = "rpn_weights.h5"
MODEL_JSON_FILE = "rpn.json"

if __name__ == "__main__":
    config = Config()

    # MODEL
    wrapper = MaskRCNNWrapper(config)
    wrapper.compile_model()

    # INPUT
    print("OBTAINING INPUT ...")
    images, rpn_cls_gt, rpn_reg_gt = utils.data_generator(config)

    # TRAINING
    print("TRAINING ...")
    wrapper.fit_model(inputs=[images, rpn_cls_gt, rpn_reg_gt])

    # TODO: Show metrics (accuracy)

    # SAVE MODEL
    # TODO: Fix saving (maybe custom layers need special treatment?)
    #wrapper.model.save(MODEL_FILE)
    #wrapper.model.save_weights(MODEL_WEIGHTS_FILE)

    # Write model to file for inspection
    #model_json = wrapper.model.to_json()
    # with open(MODEL_JSON_FILE, 'w+') as model_json_file:
    #     model_json_file.write(model_json)

