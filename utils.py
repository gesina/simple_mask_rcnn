import keras
import numpy as np
import mnist_mask as mm
import config as conf
import os
from tqdm import tqdm
from math import log, exp
import random


####################
# Further Settings #
####################

MIN_IOU = 0.3

NPY_FOLDER = "data/mask_rcnn/prepared_npy"
NPY_IMAGES_FILENAME = "images.npy"
NPY_RPN_CLS_FILENAME = "rpn_cls.npy"
NPY_RPN_CLS_TRAINING_FILENAME = "rpn_cls_train.npy"
NPY_RPN_REG_FILENAME = "rpn_reg.npy"
NPY_RPN_REG_DELTAS_FILENAME = "rpn_reg_deltas.npy"
NPY_ORIGINAL_SHAPES_FILENAME = "orig_shapes.npy"
# Output of data_from_folder as .npy file names for saving
# Mind the order!
NPY_FILENAMES = (NPY_IMAGES_FILENAME,
                 NPY_RPN_CLS_FILENAME,
                 NPY_RPN_CLS_TRAINING_FILENAME,
                 NPY_RPN_REG_FILENAME,
                 NPY_RPN_REG_DELTAS_FILENAME,
                 NPY_ORIGINAL_SHAPES_FILENAME)
OUTPUT_FILEPATH_FORMAT = "data/mask_rcnn/test/test{}.jpg"
LOGFILEPATH = "log"


#######################
# Box Transformations #
#######################

def box_to_delta(box, anchor):
    """((x1,y1) = upper left corner, (x2, y2) = lower right corner):
            * box center point, width, height = (x, y, w, h)
            * anchor center point, width, height = (x_a, y_a, w_a, h_a)
            * anchor = (x1=x_a-w_a/2, y1=y_a-h_a/2, x2=x_a+w_a/2, y2=y_a+h_a/2)
            * box = (x1=x-w/2, y1=y-h/2, x2=x+w/2, y2=y+w/2)
            * box_delta = ((x-x_a)/w_a, (y-y_a)/h_a, log(w/w_a), log(h/h_a))

    :param tuple box: box coordinates
    :param tuple anchor: anchor coordinates
    :return: box delta coordinates as described above
    """
    # box
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x, y = x1 + w / 2, y1 + h / 2

    # anchor
    x1_a, y1_a, x2_a, y2_a = anchor
    w_a, h_a = x2_a - x1_a, y2_a - y1_a
    x_a, y_a = x1_a + w_a / 2.0, y1_a + h_a / 2.0

    dx, dy = (x - x_a) / w_a, (y - y_a) / h_a
    dw, dh = log(w / w_a), log(h / h_a)

    return dx, dy, dw, dh


def delta_to_box(delta, anchor):
    """Inverse to box_to_delta:
        * box_delta = (dx, dy, log(dw), log(dh))
        * anchor center point, width, height = (x_a, y_a, w_a, h_a)
        * anchor = (x1=x_a-w_a/2, y1=y_a-h_a/2, x2=x_a+w_a/2, y2=y_a+h_a/2)
        * box center point, width, height = (x_a + dx*w_a, y_a + dy*h_a, w_a*exp(dw), h_a*exp(dh))
        * box = (x1=x-w/2, y1=y-h/2, x2=x+w/2, y2=y+w/2)

    :param tuple delta: delta values
    :param tuple anchor: anchor coordinates
    :return: box coordinates
    """
    # Deltas
    dx, dy, logdw, logdh = delta

    # Anchor
    x1_a, y1_a, x2_a, y2_a = anchor
    w_a = x2_a - x1_a
    h_a = y2_a - y1_a
    x_a = x1_a + w_a / 2.0
    y_a = y1_a + h_a / 2.0

    # Box center coordinates
    x = x_a + dx * w_a
    y = y_a + dy * h_a
    w = w_a * exp(logdw)
    h = h_a * exp(logdh)

    # Box coordinates
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x1 + w
    y2 = y1 + h

    return x1, y1, x2, y2


def box_raw_to_normalized(box, image_width, image_height):
    """Reformat box from mm format to [x1, y1, x2, y2] in normalized coordinates.

    :param int image_width: original total width of image
    :param int image_height: original total height of image
    :param tuple box: tuple as ((x1, y1), (x2, y2))
    :return: [x1, y1, x2, y2] in normalized coordinates
    """
    x1, y1 = conf.to_norm_coordinates(*box[0], total_width=image_width, total_height=image_height)
    x2, y2 = conf.to_norm_coordinates(*box[1], total_width=image_width, total_height=image_height)
    return [x1, y1, x2, y2]


def box_normalized_to_raw(box, image_width, image_height):
    """Reformat box as needed for mm lib.

    :param list box: [x1, y1, x2, y2]
    :param int image_width: width of the total image in px
    :param int image_height: height of the total image in px
    :return: box as needed for the mm lib
    """
    x1, y1, x2, y2 = box
    box = [x1 * image_width, y1 * image_height, x2 * image_width, y2 * image_height]
    x1, y1, x2, y2 = list(map(lambda z: int(round(z)), box))
    return (x1, y1), (x2, y2)


def boxes_normalized_to_raw(boxes, image_width, image_height):
    """Apply box_normalized_to_mm() to a list of boxes."""
    return list(map(
        lambda box:
        box_normalized_to_raw(box,
                              image_width=image_width,
                              image_height=image_height),
        boxes
    ))


#################################
# Data Parsing Helper Functions #
#################################

def iou(box1, box2):
    """Intersection over Union value."""
    # Intersection rectangle
    intersect_x1 = max(box1[0], box2[0])
    intersect_y1 = max(box1[1], box2[1])
    intersect_x2 = min(box1[2], box2[2])
    intersect_y2 = min(box1[3], box2[3])

    # Area of intersection rectangle
    if intersect_x1 >= intersect_x2 or intersect_y1 >= intersect_y2:
        return 0.0
    area_intersection = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)

    # Area of both boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Intersection over union ratio: intersection_area/union_area
    area_union = float(area_box1 + area_box2 - area_intersection)
    return area_intersection / area_union if area_union != 0 else 1


def generate_gt_labels(boxes, anchors,
                       max_iou_negative, min_iou_positive):
    """

    :param np.array boxes: [num_boxes, 4: (x1, y1, x2, y2)]
        bounding box coordinates in normalized coordinates
    :param np.array anchors: [num_boxes, 4: (x1, y1, x2, y2)]
        anchor box coordinates (x1, y1, x2, y2) in normalized coordinates
    :param float max_iou_negative: minimum intersection over union ratio of an anchor with an object
        bounding box below which an anchor is considered not definitely not contain that object
    :param float min_iou_positive: minimum intersection over union ratio of an anchor with an object
        bounding box above which an anchor is considered not definitely contain that object
    :return: cls_gt, reg_gt tensors as needed as rows for loss functions
        - np.array cls_gt: [num_anchors, 1: class +1/-1/0] one row of rpn_cls_gt
        - np.array reg_gt: [num_anchors, 4: box coord (x1, y1, x2, y2)] one row of rpn_reg_gt
        - np.array reg_delta_gt: [num_anchors, 4: box deltas] where the box deltas are
            encoded as described in box_to_delta()
    """
    num_anchors = anchors.shape[0]
    num_boxes = boxes.shape[0]

    # see config.RPN_???_SHAPE
    cls_gt = np.zeros(shape=[num_anchors])
    reg_gt = np.zeros(shape=[num_anchors, 4])
    reg_deltas_gt = np.zeros(shape=[num_anchors, 4])

    # calc IoUs for each box and anchor
    ious = np.zeros(shape=[num_boxes, num_anchors])
    for i in range(0, num_boxes):
        for j in range(0, num_anchors):
            ious[i, j] = iou(boxes[i], anchors[j])

    # Each box:
    #  best (unset) IoU anchor 1 (yes for this box)
    # TODO: Better algorithm to get best available IoU anchor; this one implies shifts down and to the right
    for i in range(num_boxes):
        box_max_iou_idx = np.argmax(np.where(cls_gt != 1, ious[i, :], -1))
        cls_gt[box_max_iou_idx] = 1
        reg_gt[box_max_iou_idx, :] = boxes[i]
        reg_deltas_gt[box_max_iou_idx, :] = box_to_delta(boxes[i], anchors[box_max_iou_idx])

    # Each non-set anchor (maybe a box is approximated well by more than one anchor):
    #  best IoU >= min_iou_positive? -> 1 (yes for that box)
    #  best IoU <= max_iou_negative? -> -1 (no)
    #  else -> 0 (stay neutral) with best bounding box
    # The last (0s get bounding boxes) is a test, taking in neutral anchors for
    # coordinate correction accuracy for misclassified anchor boxes.
    for j in range(0, num_anchors):
        if cls_gt.flat[j] == 0:
            anchor_max_iou_idx = np.argmax(ious[:, j])
            anchor_max_iou = ious[anchor_max_iou_idx, j]
            if anchor_max_iou >= min_iou_positive:
                cls_gt[j] = 1
                reg_gt[j, :] = boxes[anchor_max_iou_idx]
                reg_deltas_gt[j, :] = box_to_delta(boxes[anchor_max_iou_idx], anchors[j])
            elif anchor_max_iou <= max_iou_negative:
                cls_gt[j] = -1
                # reg_gt[j, :] stays zero padding
                # reg_deltas_gt[j, :] stays zero padding
            else:
                # cls_gt[j] stays 0
                reg_gt[j, :] = boxes[anchor_max_iou_idx]
                reg_deltas_gt[j, :] = box_to_delta(boxes[anchor_max_iou_idx], anchors[j])

    # Did all boxes get at least one anchor?
    testl = [i for i in range(num_boxes)
             if boxes[i].tolist() not in reg_gt.tolist()]
    assert len(testl) == 0, "Some boxes did not get an anchor: " + str([(i, boxes[i]) for i in testl])
    return cls_gt.reshape(num_anchors, 1), reg_gt, reg_deltas_gt


def randomly_balance_pos_neg_samples(cls, border_crossing_anchor_indices, balance_factor):
    """Balance the number of positive and negative objectness class samples
    by randomly setting enough negative samples to neutral.

    :param np.array cls: [num_anchors, 1: class +1/-1/0]
        ground truth values of objectness classes for one image's anchors;
            * +1: positive
            * -1: negative
            * 0: neutral (does not contribute to training loss)
    :param list border_crossing_anchor_indices: indices of the anchors that cross
    a boundary and have to be blacklisted for training (by setting their cls
    value to 0)
    :param float balance_factor: Factor to get max_num_negative_samples from num_pos_samples;
    the number of samples marked as negative is the chosen randomly between
    num_pos_samples and balance_factor * num_pos_samples
    :return: cls with only an equal amount of positive and negative samples;
        the rest of the negative samples set to neutral (0)
    """
    num_pos_samples = int(np.sum(cls == 1))
    neg_indices = np.nonzero(cls == -1)[0].tolist()

    # Indices that may be marked as negative: gt negative boxes that do not cross a border
    possible_neg_indices = [ind for ind in neg_indices
                            if ind not in border_crossing_anchor_indices]
    max_num_negative_samples = random.randint(num_pos_samples, int(round(num_pos_samples * balance_factor)))
    max_num_negative_samples = min(max_num_negative_samples, len(possible_neg_indices))

    rand_negative_indices = random.sample(possible_neg_indices, max_num_negative_samples)

    # Set all negative samples to neutral
    cls[cls == -1] = 0
    # Set max_num_negative_samples of the negative samples back to negative
    cls[rand_negative_indices] = -1

    return cls


def simplify_image(image, config):
    if config.INVERTED_COLORS:
        image = mm.invert(image)
    if config.GRAYSCALE:
        image = mm.to_grayscale(image)  # inverted grayscale is much easier to train
        image = np.reshape(image, list(image.shape) + [1])
    if config.NORMALIZE_IMAGE:
        image = image.astype(np.float32) / 255
    return image


def desimplify_image(image, config):
    """Inverse (as far as possible) to simplify_image()"""
    if config.NORMALIZE_IMAGE:
        image *= 255
        image = image.astype(np.uint8)
    if config.GRAYSCALE:
        image = image.reshape(image.shape[:-1])
        image = mm.to_bgr_colorspace(image)
    if config.INVERTED_COLORS:
        image = mm.invert(image)
    return image


###########################
# Data Loading/Generation #
###########################

def check_data(warning_message_format="WARNING: Data {} containing {} values!",
               checks=None,
               **data):
    """Apply checks on data.

    :param str warning_message_format: formatting string accepting datum-key and check-key.
    :param dict checks: dict {check_name: check_function}; default: NaN check, +/-inf check
    :param dict data: np.arrays to check entries of
    :return: True if a warning was given, else False
    """
    if checks is None:
        checks = {"NaN": np.isnan, "-inf": np.isneginf, "inf": np.isinf}
    warned = False
    for key, datum in data.items():
        for check, fn in checks.items():
            if fn(datum).any():
                print(warning_message_format.format(key, check))
                warned = True
    return warned


def data_from_folder(config):
    """Read and parse image data and metadata into Mask R-CNN model input.

    :param Config config: configuration
    :return: images, rpn_cls_gt, rpn_reg_gt as needed for input and losses
        and the list original_shapes listing the shapes of the images loaded
    """
    # TODO: make data_generator a real generator to save RAM
    images, all_matches_raw, original_shapes = mm.load_labeled_data(image_shape=config.IMAGE_SHAPE)

    train_images, rpn_cls_gt, rpn_cls_gt_training, rpn_reg_gt, rpn_reg_deltas_gt = [], [], [], [], []
    # Iterate over images
    for i in tqdm(range(0, len(images)), "Parsing image files"):
        image = simplify_image(images[i], config=config)
        raw_matches = all_matches_raw[i]
        original_shape = original_shapes[i]
        image_width, image_height = original_shape[1], original_shape[0]

        # No matches?
        if len(raw_matches) == 0:
            cls = np.zeros(shape=config.RPN_CLS_SHAPE)
            reg = np.zeros(shape=config.RPN_REG_SHAPE)
            reg_deltas = np.zeros(shape=config.RPN_REG_SHAPE)
        # Yes, there are matches:
        else:
            labels, boxes, masks = zip(*raw_matches)

            # ANCHOR CLASSES AND GROUND-TRUTH BOXES
            # Unpack boxes to np.array with absolute coordinates
            boxes = list(map(lambda box:
                             box_raw_to_normalized(box=box, image_width=image_width, image_height=image_height),
                             boxes))
            # Get objectness class and (for positive ones) ground-truth box coordinates
            # for each anchor in a list; this is one row of rpn_cls_gt, rpn_get_gt resp.
            cls, reg, reg_deltas = generate_gt_labels(np.array(boxes), np.array(config.ANCHOR_BOXES),
                                                      max_iou_negative=config.MAX_IOU_NEGATIVE,
                                                      min_iou_positive=config.MIN_IOU_POSITIVE)
        # TODO: Find more efficient way to generate repetitions with different balances
        for rep in range(0, config.NUM_BALANCED_REPETITIONS):
            train_images.append(image)
            rpn_cls_gt.append(cls)
            rpn_cls_gt_training.append(
                randomly_balance_pos_neg_samples(np.copy(cls),
                                                 border_crossing_anchor_indices=config.BOUNDARY_ANCHOR_INDICES,
                                                 balance_factor=config.BALANCE_FACTOR))
            rpn_reg_gt.append(reg)
            rpn_reg_deltas_gt.append(reg_deltas)

    train_images = np.array(train_images)
    rpn_cls_gt = np.array(rpn_cls_gt)
    rpn_cls_gt_training = np.array(rpn_cls_gt_training)
    rpn_reg_gt = np.array(rpn_reg_gt)
    rpn_reg_deltas_gt = np.array(rpn_reg_deltas_gt)
    original_shapes = np.array(original_shapes)

    return train_images, rpn_cls_gt, rpn_cls_gt_training, rpn_reg_gt, rpn_reg_deltas_gt, original_shapes


def load_maskrcnn_data(config,
                       reload=False, save_new_as_npy=True,
                       do_check_data=False,
                       npy_filenames=NPY_FILENAMES):
    npy_filepaths = [os.path.join(NPY_FOLDER, fn) for fn in npy_filenames]

    # No .npy files yet?
    if not os.path.isdir(NPY_FOLDER):
        os.makedirs(NPY_FOLDER)
    if reload or any([not os.path.isfile(f) for f in npy_filepaths]):
        data_tuple = data_from_folder(config)
        if save_new_as_npy:
            print("Saving parsed data ...")
            for i in tqdm(range(0, len(data_tuple)), ".npy files written"):
                data_tuple[i].dump(npy_filepaths[i])
        do_check_data = True  # check newly adquired data!
    else:
        print("Loading image data ...")
        data_tuple = tuple(map(lambda f: np.load(f), npy_filepaths))

    # Check data validity:
    if do_check_data:
        print("Checking data ...")
        check_data(**{npy_filenames[i].replace(".npy", ""): data_tuple[i]
                      for i in range(0, len(data_tuple))})

    return data_tuple


def load_backbone_pretraining_data(config, gen_config=mm.GenerationConfig()):
    gen_config.LETTER_RESOLUTION = config.BACKBONE_TRAINING_IMAGE_SHAPE[1], config.BACKBONE_TRAINING_IMAGE_SHAPE[0]
    (x_train, y_train, mask_train), (x_test, y_test, mask_test) = \
        mm.load_data(gen_config, do_convert_to_gray=True)

    # Normalize data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Reshape data
    x_train = x_train.reshape(x_train.shape[0], *config.BACKBONE_TRAINING_IMAGE_SHAPE)
    x_test = x_test.reshape(x_test.shape[0], *config.BACKBONE_TRAINING_IMAGE_SHAPE)

    # Categorical data
    y_train = keras.utils.to_categorical(y_train, config.NUM_BACKBONE_PRETRAINING_CLASSES)
    y_test = keras.utils.to_categorical(y_test, config.NUM_BACKBONE_PRETRAINING_CLASSES)

    return (x_train, y_train), (x_test, y_test)


##########################
# Data/Output Inspection #
##########################

def write_solutions(config,
                    input_images, bounding_boxes, image_shapes, gt_boxes=None,
                    gt_cls=None, anchors=None, reg_deltas=None,
                    cls=None, train_cls=None,
                    output_filepath_format=OUTPUT_FILEPATH_FORMAT,
                    verbose=True,
                    overwrite=True):
    """Writes out given boxes and other information for input_images to files.

    :param Config config: config containing data for image desimplifying; see desimplify_image()
    :param np.array input_images: [batch_size, height, width, channels]
        array of input images
    :param np.array bounding_boxes: [batch_size, num_anchors, 4: (x1,y1,x2,y2) normalized coord]
        array of predicted bounding boxes for each image
    :param np.array gt_boxes: [batch_size, num_anchors, 4: (x1,y1,x2,y2) normalized coord]
        array of ground-truth boxes for each image
    :param np.array gt_cls: [batch_size, num_anchors]
        array of ground-truth objectness class labels (+1=object, 0=neutral, -1=no obj.)
    :param np.array anchors: [num_anchors, 4: (x1,y1,x2,y2) normalized coord]
        coordinates of the anchor boxes (the same for each image)
    :param np.array reg_deltas: [batch_size, num_anchors, 4: (dx,dy,log(dw),log(dh)) normalized deltas]
        coordinate correction deltas for anchor boxes to apply; see delta_to_box() for details on encoding
    :param np.arary cls: [batch_size, num_anchors, 2: (bg score, fg score)]
        predicted objectness class probabilities (non-object prob. & obj. prob.)
    :param np.arary train_cls: [batch_size, num_anchors, 1: fg score]
        objectness class probabilities taken for training
        (differs from gt_cls due to balancing, see randomly_balance_pos_neg_samples)
    :param tuple image_shapes: (width, height) in px
        (original) shapes of the images to which to rescale before saving
    :param str output_filepath_format: format string that accepts the image index
        and filename to which the image is written; all folders have to exist
    :param boolean verbose: whether to print information on drawn boxes
    :param boolean overwrite: wheter to overwrite logfile
    """
    col_pred_boxes = (0, 0, 255)  # red
    col_best_anchors = (100, 100, 0)  # dark green
    thickness_best_anchors = 3
    col_pred_boxes_by_deltas = (0, 255, 0)  # green
    col_gt_boxes = (255, 0, 0)  # blue

    print("Output image files are", OUTPUT_FILEPATH_FORMAT)
    print("Writing predicted boxes:  yes")
    print("Writing predicted boxes by deltas: ",
          "yes" if anchors is not None and gt_cls is not None and reg_deltas is not None else "no")
    print("Writing ground-truth boxes: ", "yes" if gt_boxes is not None else "no")
    print("Writing best anchor boxes: ", "yes" if anchors is not None and gt_cls is not None else "no")
    print("Logging ground-truth classes to", LOGFILEPATH, ":", "yes" if gt_cls is not None else "no")
    print("Logging ground-truth training classes to", LOGFILEPATH, ":", "yes" if train_cls is not None else "no")
    print("Logging predicted classes to", LOGFILEPATH, ":", "yes" if cls is not None else "no")

    # Check input
    assert not check_data(bounding_boxes), str(bounding_boxes) + "Bounding boxes incorrect"
    assert not check_data(gt_boxes), str(gt_boxes) + "Gt_boxes incorrect"
    assert not check_data(gt_cls), str(gt_cls) + "Gt_cls incorrect"
    assert not check_data(anchors), str(anchors) + "Anchors incorrect"
    assert not check_data(reg_deltas), str(reg_deltas) + "Reg_deltas incorrect"

    # Check files
    if os.path.isfile(LOGFILEPATH):
        if overwrite:
            print("Overwriting old log file", LOGFILEPATH, ": yes")
            os.remove(LOGFILEPATH)
        else:
            raise FileExistsError("Logfile " + LOGFILEPATH + " exists!")
    log_root = os.path.dirname(os.path.abspath(LOGFILEPATH))
    if not os.path.isdir(log_root):
        os.makedirs(log_root)
    test_root = os.path.dirname(os.path.abspath(output_filepath_format))
    if not os.path.isdir(test_root):
        os.makedirs(test_root)

    image_idxs = range(0, input_images.shape[0])
    if not verbose:
        image_idxs = tqdm(image_idxs, "Writing image outputs")
    for img_idx in image_idxs:
        # Image
        shape = image_shapes[img_idx]
        image_width, image_height = shape[1], shape[0]
        img = desimplify_image(input_images[img_idx], config)
        img = mm.simple_resize(img, (image_width, image_height))

        # Predicted bounding boxes of best anchors (by box deltas)
        if anchors is not None and gt_cls is not None and reg_deltas is not None:
            curr_bounding_boxes_by_deltas = [
                box_normalized_to_raw(list(delta_to_box(reg_deltas[img_idx, i], anchors[i])),
                                      image_width=image_width, image_height=image_height)
                for i in range(0, anchors.shape[0])
                if gt_cls[img_idx, i] == 1
            ]
            img = mm.draw_bounding_boxes(img, curr_bounding_boxes_by_deltas, color=col_pred_boxes_by_deltas,
                                         thickness=thickness_best_anchors)

        # Ground-truth bounding boxes
        if gt_boxes is not None:
            curr_gt_boxes = boxes_normalized_to_raw(
                gt_boxes[img_idx].tolist(),
                image_width=image_width,
                image_height=image_height
            )
            img = mm.draw_bounding_boxes(img, curr_gt_boxes, color=col_gt_boxes)
            if verbose:
                print("Ground-truth boxes:")
                for b in set([(pt1, pt2) for (pt1, pt2) in curr_gt_boxes if pt1 != pt2]):
                    print(b)
        if verbose:
            print("Predicted boxes:")
            for b in set([(pt1, pt2) for (pt1, pt2) in curr_bounding_boxes if pt1 != pt2]):
                print(b)

        # Best anchor matches
        if gt_cls is not None and anchors is not None:
            curr_anchors = boxes_normalized_to_raw(
                anchors,
                image_width=image_width,
                image_height=image_height
            )
            good_anchors = [curr_anchors[j]
                            for j in range(0, len(anchors))
                            if gt_cls[img_idx, j] == 1]
            img = mm.draw_bounding_boxes(img, good_anchors, color=col_best_anchors)
            if verbose:
                print("Pos:", len([cls for cls in gt_cls[img_idx] if cls == 1]),
                      "Neg:", len([cls for cls in gt_cls[img_idx] if cls == -1]),
                      "Neutr:", len([cls for cls in gt_cls[img_idx] if cls == 0]))

        # Predicted bounding boxes
        curr_bounding_boxes = boxes_normalized_to_raw(
            bounding_boxes[img_idx].tolist(),
            image_width=image_width,
            image_height=image_height
        )
        img = mm.draw_bounding_boxes(img, curr_bounding_boxes, color=col_pred_boxes)

        # Write to file
        filepath = output_filepath_format.format(img_idx)
        mm.write_image(filepath, img)
        if verbose:
            print("Wrote to file", filepath)

        # Log gt/training/predicted classes
        if gt_cls is not None or train_cls is not None or cls is not None:
            cls_log = []
            num_anchors = gt_cls.shape[1] if gt_cls is not None \
                else train_cls.shape[1] if train_cls is not None \
                else cls.shape[1]
            for j in range(0, num_anchors):
                cls_log.append(
                    "\t".join([str(cls_arr[img_idx, j])
                               for cls_arr in (gt_cls, train_cls, cls)
                               if cls_arr is not None]))
            with open(LOGFILEPATH, 'a') as logfile:
                logfile.write("\n\n" + filepath + "\n".join(cls_log))
