import numpy as np
import mnist_mask as mm
import config as conf
import os
from tqdm import tqdm

MIN_IOU = 0.3

NPY_FOLDER = "data/mask_rcnn/prepared_npy"
NPY_IMAGES_FILENAME = "images.npy"
NPY_RPN_CLS_FILENAME = "rpn_cls.npy"
NPY_RPN_REG_FILENAME = "rpn_reg.npy"
NPY_ORIGINAL_SHAPES_FILENAME = "orig_shapes.npy"
OUTPUT_FILENAME_FORMAT = "test{}.jpg"


def iou(box1, box2):
    """Intersection over Union value."""
    # Intersection rectangle
    intersect_x1 = max(box1[0], box2[0])
    intersect_y1 = max(box1[1], box2[1])
    intersect_x2 = min(box1[2], box2[2])
    intersect_y2 = min(box1[3], box2[3])

    # Area of intersection rectangle
    area_intersection = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)

    # Area of both boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Intersection over union ratio: intersection_area/union_area
    return area_intersection / float(area_box1 + area_box2 - area_intersection)


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
    """
    num_anchors = anchors.shape[0]
    num_boxes = boxes.shape[0]

    # see config.RPN_???_SHAPE
    cls_gt = np.zeros(shape=[num_anchors, 1])
    reg_gt = np.zeros(shape=[num_anchors, 4])

    # calc IoUs for each box and anchor
    ious = np.zeros(shape=[num_boxes, num_anchors])
    for i in range(0, num_boxes):
        for j in range(0, num_anchors):
            ious[i, j] = iou(boxes[i], anchors[j])

    # Each box:
    #  no IoU better than min_iou_positive? -> best IoU anchor 1 (yes for this box)
    for i in range(0, num_boxes):
        box_max_iou_idx = np.argmax(ious[i, :])
        if ious[i, box_max_iou_idx] < max_iou_negative:
            cls_gt[box_max_iou_idx] = 1
            reg_gt[box_max_iou_idx, :] = boxes[i]

    # Each non-marked anchor:
    #  best IoU >= min_iou_positive? -> 1 (yes for that box)
    #  best IoU < min_iou_neutral? -> -1 (no)
    #  else -> 0 (stay neutral)
    for j in range(0, num_anchors):
        if cls_gt[j] == 0:
            anchor_max_iou_idx = np.argmax(ious[:, j])
            anchor_max_iou = ious[anchor_max_iou_idx, j]
            if anchor_max_iou >= min_iou_positive:
                cls_gt[j] = 1
                reg_gt[j, :] = boxes[anchor_max_iou_idx]
            elif anchor_max_iou <= max_iou_negative:
                cls_gt[j] = -1

    return cls_gt, reg_gt


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

    rpn_cls_gt, rpn_reg_gt = [], []
    # Iterate over images
    for i in tqdm(range(0, len(images)), "Parsing image files:"):
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
            cls, reg = to_rpn_gt(np.array(boxes), np.array(config.ANCHOR_BOXES),
                                 max_iou_negative=config.MAX_IOU_NEGATIVE,
                                 min_iou_positive=config.MIN_IOU_POSITIVE)
        rpn_cls_gt.append(cls)
        rpn_reg_gt.append(reg)

    images = np.array(images)
    rpn_cls_gt = np.array(rpn_cls_gt)
    rpn_reg_gt = np.array(rpn_reg_gt)
    original_shapes = np.array(original_shapes)
    return images, rpn_cls_gt, rpn_reg_gt, original_shapes


def data_generator(config):
    # TODO: Reduce code replications
    # npy_filenames = [NPY_IMAGES_FILENAME, NPY_RPN_CLS_FILENAME, NPY_RPN_REG_FILENAME]
    # npy_filepaths = [os.path.join(NPY_FOLDER, fn) for fn in npy_filenames]
    images_npy = os.path.join(NPY_FOLDER, NPY_IMAGES_FILENAME)
    rpn_cls_npy = os.path.join(NPY_FOLDER, NPY_RPN_CLS_FILENAME)
    rpn_reg_npy = os.path.join(NPY_FOLDER, NPY_RPN_REG_FILENAME)
    original_shapes_npy = os.path.join(NPY_FOLDER, NPY_ORIGINAL_SHAPES_FILENAME)

    # No .npy files yet?
    if not os.path.isdir(NPY_FOLDER):
        os.makedirs(NPY_FOLDER)
    # if any([not os.path.isfile(f) for f in npy_filepaths]):
    if not os.path.isfile(images_npy) or \
            not os.path.isfile(rpn_cls_npy) or \
            not os.path.isfile(rpn_reg_npy) or \
            not os.path.isfile(original_shapes_npy):
        images, rpn_cls_gt, rpn_reg_gt, original_shapes = data_from_folder(config)
        images.dump(images_npy)
        rpn_cls_gt.dump(rpn_cls_npy)
        rpn_reg_gt.dump(rpn_reg_npy)
        original_shapes.dump(original_shapes_npy)
    else:
        print("Loading image data ...")
        images, rpn_cls_gt, rpn_reg_gt, original_shapes = \
            np.load(images_npy), np.load(rpn_cls_npy), np.load(rpn_reg_npy), np.load(original_shapes_npy)

    return images, rpn_cls_gt, rpn_reg_gt, original_shapes


def write_solutions(input_images, bounding_boxes, image_shapes, gt_boxes=None,
                    output_filename_format=OUTPUT_FILENAME_FORMAT,
                    verbose=True):
    """Writes out bounding_box solutions for input_images to files.

    :param np.array input_images: array of input images
    :param np.array bounding_boxes: array of input boxes for each image
    :param np.array gt_boxes: array of ground-truth boxes for each image if available
    :param tuple image_shapes: (width, height) in px
    :param str output_filename_format: format string that accepts the image index
        and filename to which the image is written.
    :param boolean verbose: whether to print information on drawn boxes
    """
    for i in range(0, input_images.shape[0]):
        # Image
        shape = image_shapes[i]
        image_width, image_height = shape[1], shape[0]
        img = mm.simple_resize(input_images[i], (image_width, image_height))

        # Predicted bounding boxes
        curr_bounding_boxes = boxes_normalized_to_raw(
            bounding_boxes[i].tolist(),
            image_width=image_width,
            image_height=image_height
        )
        img = mm.draw_bounding_boxes(img, curr_bounding_boxes, color=mm.BOUNDING_BOX_COLOR)

        # Ground-truth bounding boxes
        if gt_boxes is not None:
            curr_gt_boxes = boxes_normalized_to_raw(
                gt_boxes[i].tolist(),
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

        # Write to file
        filepath = output_filename_format.format(i)
        mm.write_image(filepath, img)
        if verbose:
            print("Wrote to file", filepath)
