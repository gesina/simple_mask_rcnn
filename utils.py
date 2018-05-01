import numpy as np
import mnist_mask as mm
import config as conf
import os
from tqdm import tqdm
from math import log

MIN_IOU = 0.3

NPY_FOLDER = "data/mask_rcnn/prepared_npy"
NPY_IMAGES_FILENAME = "images.npy"
NPY_RPN_CLS_FILENAME = "rpn_cls.npy"
NPY_RPN_REG_FILENAME = "rpn_reg.npy"
NPY_RPN_REG_DELTAS_FILENAME = "rpn_reg_deltas.npy"
NPY_ORIGINAL_SHAPES_FILENAME = "orig_shapes.npy"
# Output of data_from_folder as .npy file names for saving
# Mind the order!
NPY_FILENAMES = (NPY_IMAGES_FILENAME,
                 NPY_RPN_CLS_FILENAME,
                 NPY_RPN_REG_FILENAME,
                 NPY_RPN_REG_DELTAS_FILENAME,
                 NPY_ORIGINAL_SHAPES_FILENAME)
OUTPUT_FILEPATH_FORMAT = "data/mask_rcnn/test/test{}.jpg"


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
    x_a, y_a = x1_a + w_a / 2, y1_a + h_a / 2

    dx, dy = (x - x_a) / w_a, (y - y_a) / h_a
    dw, dh = log(w / w_a), log(h / h_a)

    return dx, dy, dw, dh


def to_rpn_gt(boxes, anchors,
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
    cls_gt = np.zeros(shape=[num_anchors, 1])
    reg_gt = np.zeros(shape=[num_anchors, 4])
    reg_deltas_gt = np.zeros(shape=[num_anchors, 4])

    # calc IoUs for each box and anchor
    ious = np.zeros(shape=[num_boxes, num_anchors])
    for i in range(0, num_boxes):
        for j in range(0, num_anchors):
            ious[i, j] = iou(boxes[i], anchors[j])

    # Each anchor:
    #  best IoU >= min_iou_positive? -> 1 (yes for that box)
    #  best IoU <= max_iou_negative? -> -1 (no)
    #  else -> 0 (stay neutral)
    for j in range(0, num_anchors):
        anchor_max_iou_idx = np.argmax(ious[:, j])
        anchor_max_iou = ious[anchor_max_iou_idx, j]
        if anchor_max_iou >= min_iou_positive:
            cls_gt[j] = 1
            reg_gt[j, :] = boxes[anchor_max_iou_idx]
            reg_deltas_gt[j, :] = box_to_delta(boxes[anchor_max_iou_idx], anchors[j])
        elif anchor_max_iou <= max_iou_negative:
            cls_gt[j] = -1

    # Each box:
    #  no IoU better than min_iou_positive? -> best IoU anchor 1 (yes for this box)
    for i in range(0, num_boxes):
        box_max_iou_idx = np.argmax(ious[i, :])
        if cls_gt[box_max_iou_idx] != 1:
            cls_gt[box_max_iou_idx] = 1
            reg_gt[box_max_iou_idx, :] = boxes[i]
            reg_deltas_gt[box_max_iou_idx, :] = box_to_delta(boxes[i], anchors[box_max_iou_idx])

    # Did all boxes get at least one anchor?
    assert len([i for i in range(0, num_boxes)
                if boxes[i].tolist() not in reg_gt.tolist()]) == 0
    return cls_gt, reg_gt, reg_deltas_gt


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


def data_from_folder(config):
    """Read and parse image data and metadata into Mask R-CNN model input.

    :param Config config: configuration
    :return: images, rpn_cls_gt, rpn_reg_gt as needed for input and losses
        and the list original_shapes listing the shapes of the images loaded
    """
    # TODO: make data_generator a real generator to save RAM
    images, all_matches_raw, original_shapes = mm.load_labeled_data(image_shape=config.IMAGE_SHAPE)

    rpn_cls_gt, rpn_reg_gt, rpn_reg_deltas_gt = [], [], []
    # Iterate over images
    for i in tqdm(range(0, len(images)), "Parsing image files"):
        raw_matches = all_matches_raw[i]
        original_shape = original_shapes[i]
        image_width, image_height = original_shape[1], original_shape[0]

        # No matches?
        if len(raw_matches) == 0:
            cls = np.zeros(shape=config.RPN_CLS_SHAPE)
            reg = np.zeros(shape=config.RPN_REG_SHAPE)
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
            cls, reg, reg_deltas = to_rpn_gt(np.array(boxes), np.array(config.ANCHOR_BOXES),
                                             max_iou_negative=config.MAX_IOU_NEGATIVE,
                                             min_iou_positive=config.MIN_IOU_POSITIVE)
        rpn_cls_gt.append(cls)
        rpn_reg_gt.append(reg)
        rpn_reg_deltas_gt.append(reg_deltas)

    images = np.array(images)
    rpn_cls_gt = np.array(rpn_cls_gt)
    rpn_reg_gt = np.array(rpn_reg_gt)
    rpn_reg_deltas_gt = np.array(rpn_reg_deltas_gt)
    original_shapes = np.array(original_shapes)
    return images, rpn_cls_gt, rpn_reg_gt, rpn_reg_deltas_gt, original_shapes


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


def data_generator(config, do_check_data=False,
                   npy_filenames=NPY_FILENAMES):
    # TODO: Reduce code replications
    npy_filepaths = [os.path.join(NPY_FOLDER, fn) for fn in npy_filenames]

    # No .npy files yet?
    if not os.path.isdir(NPY_FOLDER):
        os.makedirs(NPY_FOLDER)
    if any([not os.path.isfile(f) for f in npy_filepaths]):
        data_tuple = data_from_folder(config)
        print("Saving parsed data ...")
        for i in range(0, len(data_tuple)):
            data_tuple[i].dump(npy_filepaths[i])
        # images, rpn_cls_gt, rpn_reg_gt, rpn_reg_deltas_gt, original_shapes = data_from_folder(config)
        # images.dump(images_npy)
        # rpn_cls_gt.dump(rpn_cls_npy)
        # rpn_reg_gt.dump(rpn_reg_npy)
        # original_shapes.dump(original_shapes_npy)
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


def write_solutions(input_images, bounding_boxes, image_shapes, gt_boxes=None,
                    gt_cls=None, anchors=None,
                    output_filepath_format=OUTPUT_FILEPATH_FORMAT,
                    verbose=True):
    """Writes out bounding_box solutions for input_images to files.

    :param np.array input_images: array of input images
    :param np.array bounding_boxes: array of input boxes for each image
    :param np.array gt_boxes: array of ground-truth boxes for each image if available
    :param tuple image_shapes: (width, height) in px
    :param str output_filepath_format: format string that accepts the image index
        and filename to which the image is written; all folders have to exist
    :param boolean verbose: whether to print information on drawn boxes
    """
    image_idxs = range(0, input_images.shape[0])
    if not verbose:
        image_idxs = tqdm(image_idxs, "Writing image outputs")
    for img_idx in image_idxs:
        # Image
        shape = image_shapes[img_idx]
        image_width, image_height = shape[1], shape[0]
        img = mm.simple_resize(input_images[img_idx], (image_width, image_height))

        # Predicted bounding boxes
        curr_bounding_boxes = boxes_normalized_to_raw(
            bounding_boxes[img_idx].tolist(),
            image_width=image_width,
            image_height=image_height
        )
        img = mm.draw_bounding_boxes(img, curr_bounding_boxes, color=mm.BOUNDING_BOX_COLOR)

        # Ground-truth bounding boxes
        if gt_boxes is not None:
            curr_gt_boxes = boxes_normalized_to_raw(
                gt_boxes[img_idx].tolist(),
                image_width=image_width,
                image_height=image_height
            )
            img = mm.draw_bounding_boxes(img, curr_gt_boxes, color=mm.MAX_BOUNDING_BOX_COLOR)
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
            img = mm.draw_bounding_boxes(img, good_anchors, color=(255, 0, 0))
            if verbose:
                print("Pos:", len([cls for cls in gt_cls[img_idx] if cls == 1]),
                      "Neg:", len([cls for cls in gt_cls[img_idx] if cls == -1]),
                      "Neutr:", len([cls for cls in gt_cls[img_idx] if cls == -1]))

        # Write to file
        filepath = output_filepath_format.format(img_idx)
        mm.write_image(filepath, img)
        if verbose:
            print("Wrote to file", filepath)
