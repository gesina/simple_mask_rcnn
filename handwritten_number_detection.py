#! /bin/python3

from model import MaskRCNNWrapper
from config import Config
import utils
import os

MODEL_FILE = "rpn.h5"
MODEL_WEIGHTS_FILE = "rpn_weights.h5"
MODEL_JSON_FILE = "rpn.json"

NUM_TEST_IMAGES = 100

if __name__ == "__main__":
    config = Config()

    # TRAINING MODEL
    print("SETTING UP TRAINING MODEL ...")
    train_wrapper = MaskRCNNWrapper(config)
    # Reuse pretrained weights if possible:
    if os.path.isfile(MODEL_WEIGHTS_FILE):
        train_wrapper.model.load_weights(MODEL_WEIGHTS_FILE)
    print("COMPILING TRAINING MODEL ...")
    train_wrapper.compile_model()

    # INPUT
    print("OBTAINING INPUT ...")
    images, rpn_cls_gt, rpn_reg_gt, rpn_reg_deltas_gt, original_shapes = utils.data_generator(config)

    # TRAINING
    print("TRAINING ...")
    train_wrapper.fit_model(inputs=[images, rpn_cls_gt, rpn_reg_deltas_gt])
    # SAVE MODEL
    train_wrapper.model.save_weights(MODEL_WEIGHTS_FILE)
    # TODO: train_wrapper.model.save(MODEL_FILE)
    # # Write model to file for inspection
    # model_json = wrapper.model.to_json()
    # with open(MODEL_JSON_FILE, 'w+') as model_json_file:
    #     model_json_file.write(model_json)

    # TESTING
    print("SETTING UP TEST MODEL ...")
    test_wrapper = MaskRCNNWrapper(config, train=False)
    # The model only differs by custom layers without weight,
    # thus no `by_name=True` necessary.
    print("LOADING OLD TRAINED WEIGHTS ...")
    test_wrapper.model.load_weights(MODEL_WEIGHTS_FILE)

    # test_input = utils.get_test_data([TEST_FILE], config.IMAGE_SHAPE)
    num_test_images = min(len(images), NUM_TEST_IMAGES)
    test_input = images[0:num_test_images]
    test_shapes = original_shapes[0:num_test_images]
    test_output = test_wrapper.predict(input_images=test_input)
    test_reg_gt = rpn_reg_gt[0:num_test_images]
    test_cls_gt = rpn_cls_gt[0:num_test_images]
    print(config.ANCHOR_BOXES[0])
    utils.write_solutions(test_input, test_output, test_shapes, test_reg_gt,
                          gt_cls=rpn_cls_gt, anchors=config.ANCHOR_BOXES,
                          verbose=False)
