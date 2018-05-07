#! /bin/python3

from model import MaskRCNNWrapper
from config import Config
import utils
import os

MODEL_FILE = "rpn.h5"
MODEL_WEIGHTS_FILE = "rpn_weights.h5"
BACKBONE_WEIGHTS_FILE = "backbone_weights.h5"
MODEL_JSON_FILE = "rpn.json"
NUM_TEST_IMAGES = 100


if __name__ == "__main__":
    # Script setup:
    load_training_data = True
    reload_training_data = False

    pretraining = True
    training = True
    load_backbone_weights = True
    load_weights = True
    testing = True

    if pretraining or training or testing:
        config = Config()

    if pretraining or training:
        print("\n>SETTING UP TRAINING MODELS ...")
        train_wrapper = MaskRCNNWrapper(config)

    # PRETRAINING
    if pretraining:
        print("\n>COMPILING BACKBONE PRETRAINING MODEL ...")
        train_wrapper.compile_backbone_pretraining_model()
        if os.path.isfile(BACKBONE_WEIGHTS_FILE):
            print("Using pretrained weights from", BACKBONE_WEIGHTS_FILE, "for pretraining backbone")
            train_wrapper.backbone_pretraining_model.load_weights(BACKBONE_WEIGHTS_FILE)

        print("\n>OBTAINING PRETRAINING INPUT ...")
        (x_train, y_train), (x_test, y_test) = \
            utils.load_backbone_pretraining_data(config)

        print("\n>TRAINING BACKBONE MODEL ...")
        train_wrapper.fit_backbone_training_model(inputs=x_train, labels=y_train,
                                                  validation_data=(x_test, y_test))
        score = train_wrapper.backbone_pretraining_model.evaluate(x_test, y_test, verbose=0)
        print("Pretraining results: Test loss:", score[0], "\tTest accuracy:", score[1])

        train_wrapper.backbone_pretraining_model.save_weights(BACKBONE_WEIGHTS_FILE)

    # INPUT
    if load_training_data or training or testing:
        print("\n>OBTAINING TRAINING INPUT ...")
        images, rpn_cls_gt, rpn_cls_gt_training, rpn_reg_gt, rpn_reg_deltas_gt, original_shapes = \
            utils.load_maskrcnn_data(config, reload=reload_training_data)

    if training and (load_weights or load_backbone_weights):
        # LOAD WEIGHTS
        print("\n>LOADING PRETRAINED WEIGHTS ...")
        msg = ""
        if load_backbone_weights:
            if os.path.isfile(BACKBONE_WEIGHTS_FILE):
                print("Using pretrained backbone weights from", BACKBONE_WEIGHTS_FILE)
                # Load pretrained backbone weights only with by_name=True
                train_wrapper.model.load_weights(BACKBONE_WEIGHTS_FILE, by_name=True)
            else:
                print("No backbone weights file", MODEL_WEIGHTS_FILE, " found.")
        if load_weights:
            if os.path.isfile(MODEL_WEIGHTS_FILE):
                print("Using pretrained weights from ", MODEL_WEIGHTS_FILE)
                train_wrapper.model.load_weights(MODEL_WEIGHTS_FILE)
            else:
                print("No weights file", MODEL_WEIGHTS_FILE, " found.")

    if training:
        print("\n>COMPILING TRAINING MODEL ...")
        train_wrapper.compile_model()
        # TRAINING
        print("\n>TRAINING ...")
        train_wrapper.fit_model(inputs=[images, rpn_cls_gt_training, rpn_reg_deltas_gt], 
                                checkpt_filepath_format="rpn_{epoch}.h5")
        # SAVE MODEL
        train_wrapper.model.save_weights(MODEL_WEIGHTS_FILE)

    # TESTING
    if testing:
        print("\n>SETTING UP TEST MODEL ...")
        test_wrapper = MaskRCNNWrapper(config, train=False)
        # The model only differs by custom layers without weight,
        # thus no `by_name=True` necessary.
        print("\n>LOADING OLD TRAINED WEIGHTS ...")
        test_wrapper.model.load_weights(MODEL_WEIGHTS_FILE)

        print("\n>INFERENCE ...")
        num_test_images = min(len(images), NUM_TEST_IMAGES)
        test_input = images[0:num_test_images]
        test_shapes = original_shapes[0:num_test_images]
        test_reg_output, test_reg_deltas_output, test_cls_output = test_wrapper.predict(input_images=test_input)
        test_reg_gt = rpn_reg_gt[0:num_test_images]
        test_cls_gt = rpn_cls_gt[0:num_test_images]
        print("\n>WRITING TEST OUTPUT ...")
        utils.write_solutions(config,
                              test_input, test_reg_output, test_shapes, test_reg_gt,
                              gt_cls=rpn_cls_gt, anchors=config.ANCHOR_BOXES,
                              reg_deltas=test_reg_deltas_output,
                              cls=test_cls_output, train_cls=rpn_cls_gt_training,
                              verbose=False)
