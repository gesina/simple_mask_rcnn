import numpy as np


def to_abs_coordinates(x, y, shape):
    return x / shape[1], y / shape[0]


def get_center_points(center_point_width, center_point_height, col_wise=False):
    """Create list of center points of boxes of size center_point_width x center_point_height.

    :param int center_point_width, int center_point_height: box size as [w, h] in normalized coordinates
        of the box that an upscaled center point (pixel) would take up in the image
    :param boolean col_wise: whether to join the colums to a list instead of the rows
    :return: list row-wise (resp. colum-wise) center points as (x, y)
    """
    num_center_pt_cols = int(round(1 / center_point_width))
    num_center_pt_rows = int(round(1 / center_point_height))

    center_pts = np.empty(shape=[num_center_pt_rows, num_center_pt_cols, 2])

    # Create points
    for row in range(0, num_center_pt_rows):
        for col in range(0, num_center_pt_cols):
            center_pts[row, col, 0] = (col + 0.5) * center_point_width   # x
            center_pts[row, col, 1] = (row + 0.5) * center_point_height  # y

    # Concatenate;
    # If col_wise concatenation is required, transpose first
    if col_wise:
        center_pts = list(center_pts.T)
    return np.concatenate(center_pts, axis=0)


def get_anchor_boxes(center_points, anchor_shapes):
    """

    :param List[Tuple[float, float]] center_points: list of center points (x, y)
        in normalized coordinates for the anchors
    :param list anchor_shapes: list of anchor shapes as (w, h) in normalized coordinates
    :return: anchor box coordinates as [x1, y1, x2, y2] in normalized coordinates
    """
    return [
        [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        for (x, y) in center_points
        for (w, h) in anchor_shapes
    ]


# DEFAULTS
class Config(object):
    """Configuration parameters. To adapt, create sub-class and overwrite."""

    ###################
    # TRAINING CONFIG #
    ###################
    BATCH_SIZE = 32
    EPOCHS = 3
    VALIDATION_SPLIT = 0.1

    ####################
    # FINE TUNE CONFIG #
    ####################
    # Number of proposals to be processed for masks;
    # !! Should be higher than the the maximum number of objects with bounding boxes per image !!
    NUM_PROPOSALS = 14

    # number of proposals to be processed before NMS,
    # selected by best foreground score
    PRE_NMS_LIMIT = 30

    # input image shape: [height, width, channels]
    # height, width have to be divisible by 2**3!
    IMAGE_SHAPE = [256, 256, 3]

    # What window size does one input pixel for the RPN region
    # and the RPN class network correspond to?
    # BACKBONE_DOWNSCALE * RPN_SHARED_DOWNSCALE
    # !! Must not exceed any of the image dimensions !!
    DOWNSCALING_FACTOR = 1 * 2 ** 3

    RPN_CLS_LOSS_NAME = "rpn_cls_loss"
    RPN_REG_LOSS_NAME = "rpn_reg_loss"
    LOSS_LAYER_NAMES = [
        RPN_CLS_LOSS_NAME,
        RPN_REG_LOSS_NAME
    ]

    #######################################
    # EMPIRICAL DATA                      #
    # copied from original implementation #
    #######################################
    # Maximum iou value of an anchor with an object's bounding box
    # below which the anchor is considered not to contain the object.
    MAX_IOU_NEGATIVE = 0.3

    # Minimum iou value of an anchor with an object's bounding box
    # above which the anchor is considered to definitely contain the object.
    MIN_IOU_POSITIVE = 0.7

    LOSS_WEIGHTS = {
        RPN_CLS_LOSS_NAME: 1,
        RPN_REG_LOSS_NAME: 1
    }
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    GRADIENT_CLIP_NORM = 5.0
    METRICS = ['accuracy']

    def __init__(self):
        # Threshold of foreground score for Non-Maximum Suppression
        self.NMS_THRESHOLD = self.MIN_IOU_POSITIVE

        # number of different anchor shapes per center
        self.ANCHOR_SHAPES = [
            to_abs_coordinates(28, 18, self.IMAGE_SHAPE),
            to_abs_coordinates(45, 30, self.IMAGE_SHAPE),
            to_abs_coordinates(60, 40, self.IMAGE_SHAPE)
        ]
        self.NUM_ANCHOR_SHAPES = len(self.ANCHOR_SHAPES)

        # Get center points (normalized coordinates)
        # A center point corresponds to a quadratic box of width DOWNSCALING_FACTOR.
        # From this calculate the normalized center point coordinates.
        center_pt_width = self.DOWNSCALING_FACTOR / self.IMAGE_SHAPE[1]
        center_pt_height = self.DOWNSCALING_FACTOR / self.IMAGE_SHAPE[0]
        # list of (x, y) in normalized coordinates
        self.CENTER_POINTS = get_center_points(center_point_height=center_pt_height,
                                               center_point_width=center_pt_width)

        # Get anchor box coordinates (normalized coordinates)
        self.ANCHOR_BOXES = get_anchor_boxes(self.CENTER_POINTS, self.ANCHOR_SHAPES)

        # Input shapes for the RPN network: objectness class and region coordinates
        self.RPN_CLS_SHAPE = [len(self.ANCHOR_BOXES), 1]
        self.RPN_REG_SHAPE = [len(self.ANCHOR_BOXES), 4]

        # VALIDITY CHECKS
        self.validity_checks()

    def validity_checks(self):
        """Check validity of given entries for prototyping."""
        assert self.DOWNSCALING_FACTOR <= self.IMAGE_SHAPE[0] and \
            self.DOWNSCALING_FACTOR <= self.IMAGE_SHAPE[1]

        print(len(self.CENTER_POINTS))

