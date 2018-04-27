import numpy as np
import mnist_mask as mm
import config as conf

MIN_IOU = 0.3


def iou(box1, box2):
    """Intersection over Union value."""
    # Intersection rectangle
    intersect_x1 = max(box1[0], box2[0])
    intersect_y1 = max(box1[1], box2[1])
    intersect_x2 = min(box1[2], box2[2])
    intersect_y2 = min(box1[3], box2[3])

    # Area of intersection rectangle
    area_intersection = (intersect_x2 - intersect_x1 + 1) * (intersect_y2 - intersect_y1 + 1)

    # Area of both boxes
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

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


def box_raw_to_normalized(image_shape, box):
    """

    :param image_shape: list as [height, width, channels]
    :param box: tuple as ((x1, y1), (x2, y2))
    :return: [x1, y1, x2, y2] in normalized coordinates
    """
    x1, y1 = conf.to_abs_coordinates(*box[0], image_shape)
    x2, y2 = conf.to_abs_coordinates(*box[1], image_shape)
    return [x1, y1, x2, y2]


def data_generator(config):
    """Read and parse image data and metadata into Mask R-CNN model input.

    :param Config config: configuration
    :return: images, rpn_cls_gt, rpn_reg_gt as needed for input and losses
    """
    # TODO: make data_generator a real generator to save RAM
    images, all_matches_raw = mm.load_labeled_data(image_shape=config.IMAGE_SHAPE)

    print("PARSING IMAGE DATA ...")
    rpn_cls_gt, rpn_reg_gt = [], []
    # Iterate over images
    for i in range(0, len(images)):
        print(i)
        raw_matches = all_matches_raw[i]
        # No matches?
        if len(raw_matches) == 0:
            cls = np.zeros(shape=config.RPN_CLS_SHAPE)
            reg = np.zeros(shape=config.RPN_REG_SHAPE)
        # Yes, there are matches:
        else:
            labels, boxes, masks = zip(*raw_matches)

            # ANCHOR CLASSES AND GROUND-TRUTH BOXES
            # Unpack boxes to np.array with absolute coordinates
            boxes = list(map(lambda box: box_raw_to_normalized(box=box, image_shape=config.IMAGE_SHAPE),
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
    print("DTYPES: ", images.dtype, rpn_reg_gt.dtype, rpn_cls_gt.dtype)
    print("SHAPES: ", images.shape, rpn_reg_gt.shape, rpn_cls_gt.shape)
    return images, rpn_cls_gt, rpn_reg_gt
