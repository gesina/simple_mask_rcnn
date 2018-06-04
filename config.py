import json

import numpy as np
import os
from datetime import datetime


#######################
# COMMON FUNCTIONALITY
#######################
def timestamp():
    """Get the current time as string fit for filenames."""
    return datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")


class DictlikeConfig:
    """Base class for configurations that provide dict-like update(), __str__(), and JSON dump."""
    CONFIG_STORE_FILEPATH_FORMAT = "config_{}.json"

    def to_dict(self):
        """Get all attributes as dictionary, e.g. for serialization."""
        return {
            attr: getattr(self, attr)
            for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")
        }

    def update(self, other_dict):
        """Add or update attributes as in a dictionary.

        :param dict other_dict: dictionary of the form {attribute: (new) value}
        :return: this updated object
        """
        for k, v in other_dict.items():
            setattr(self, k, v)
        return self

    def to_json_file(self, filepath=None):
        """Dump all attributes as json into filepath.

        :param str filepath: path to file where to dump the json content into;
        if None, self.CONFIG_STORE_FILEPATH is taken;
        if this is also None, a warning is printed and no action taken.
        """
        if filepath is None:
            filepath = self.CONFIG_STORE_FILEPATH_FORMAT.format(timestamp())
        if filepath is not None:
            with open(filepath, 'w+') as f:
                print("Dumping generation config into file ", filepath)
                json.dump(self.to_dict(), f)
        else:
            print("WARNING: GenerationConfig object not dumped, as no filepath was given.")

    def __str__(self):
        return '\n'.join([str(k) + "\t:\t" + str(v) for k, v in self.to_dict().items()])


#########################
# DATA GENERATION CONFIG
#########################
class GenerationConfig(DictlikeConfig):
    """Configuration settings for image generation.

    To alter defaults, create a subclass.
    To alter specific settings of an instance, use the update method below.
    """
    # Print (more) debugging messages
    DEBUG = False

    # CROP and MASK GENERATION
    MNIST_PNG_ROOT = "data/mnist_png"
    MNIST_CROP_ROOT = "data/mnist_crop"
    MNIST_MASK_ROOT = "data/mnist_mask"
    TRAIN_FOLDER = "training"
    TEST_FOLDER = "testing"

    INVERTED = True
    THRESHOLD = 100
    BOUNDING_BOX_COLOR = (0, 255, 0)
    MAX_BOUNDING_BOX_COLOR = (0, 0, 255)

    MIN_NUM_LETTERS = 0  # Minimal number of letters per image
    MAX_NUM_LETTERS = 10  # Maximum number of letters per image
    MIN_LETTERHEIGHT = 30  # Minimum size of a letter box in px
    MAX_LETTERHEIGHT = 45  # Maximum size of a letter box in px

    # How much the maximum letter size decreases per number of letters per image
    MAX_LETTERSIZE_DECREASE_FACTOR = 1
    # (Positive) Maximum number of pixels up to which letter boxes may overlap
    MAX_OVERLAP = 0
    LETTER_RESOLUTION = (28, 28)
    IMAGE_RESOLUTION = (256, 256)  # width, height

    # JSON KEYS
    JSON_MASK_KEY = "match"
    JSON_MATCHES_KEY = "matches"
    JSON_LABEL_KEY = "label"
    JSON_BOUNDING_BOX_KEY = "bounding_box"
    JSON_FILENAME_KEY = "filepath"

    JSON_MASK_RESOLUTION = (14, 14)  # compare letter resolution
    # Path to the root folder where the json files with annotations should be saved in
    DATA_ANNOTATIONSROOT = "data/mask_rcnn/annotations"
    # Path to the root folder where the image files should be saved in
    DATA_IMAGEROOT = "data/mask_rcnn/images"
    # Format string accepting the image ID for naming the image files;
    # needs a proper file ending
    IMAGE_FILENAME_FORMAT = "{}.jpg"
    # Format string accepting the batch_id with proper ending
    # for naming the annotation files
    ANNOTATIONS_FILENAME_FORMAT = "annotations00{}.json"
    # The path for a JSON-dump of an GenerationConfig object is
    # DATA_ANNOTATIONSROOT/CONFIG_STORE_FILENAME resp. None in case the last one is None
    CONFIG_STORE_FILENAME_FORMAT = "generation_config_{}.json"

    # Number of batches to perform;
    # In generate_labeled_data_files() each batch produces
    # one annotation file under annotationsroot
    NUM_BATCHES = 200
    # Number of images that are created and annotated in one file per batch
    BATCH_SIZE = 50
    # The filenames of the images serve as id, and are
    # enumerated sequentially starting at START_ID_ENUMERATION
    # in each generation process
    START_ID_ENUMERATION = 1

    # IMAGE GENERATION
    TEXTURES_FOLDER = None  # "data/textures"
    GRIDS_FOLDER = None  # "data/grids"
    STAINS_FOLDER = None  # "data/stains"

    # Maximum brightness of all letter channels together in [0, 3*255];
    # without effect if set to 0
    MAX_BRIGHTNESS = 0  # 200
    # The letters will be added to the image with a random weight in [1-max_alpha, 1];
    # without effect if set to 0 (then the weights are simply 1:1)
    MAX_ALPHA = 0.  # 0.2
    MAX_HEIGHT_VARIANCE = 0.1

    def __init__(self):
        if self.CONFIG_STORE_FILENAME_FORMAT is not None:
            self.CONFIG_STORE_FILEPATH_FORMAT = os.path.join(
                self.DATA_ANNOTATIONSROOT, self.CONFIG_STORE_FILENAME_FORMAT)
        else:
            self.CONFIG_STORE_FILEPATH_FORMAT = None

        # Make sure our default value is already calculated
        GenerationConfig.valid_colors(self.MAX_BRIGHTNESS)

    _VALID_COLORS_DICT = {}

    @staticmethod
    def valid_colors(max_brightness):
        """Give a list of all valid colors of brightness at most max_brightness.

        To save computation time, a once calculated list for a value of max_brightness
        is saved into GenerationConfig._VALID_COLORS_DICT[max_brightness] and reused
        from there if queried again.

        :param int max_brightness: maximum value of the sum of all channels;
        has to be in [0, 3*255]
        :return: List of all valid RGB colors with the sum of their channels
        being smaller than max_brightness
        """
        if max_brightness not in GenerationConfig._VALID_COLORS_DICT:
            GenerationConfig._VALID_COLORS_DICT[max_brightness] = tuple(
                ((r, g, b) for r in range(0, 255)
                 for g in range(0, 255)
                 for b in range(0, 255)
                 if r + g + b < max_brightness)
            )
        return GenerationConfig._VALID_COLORS_DICT[max_brightness]


#########################
# LEARNING CONFIG
#########################
def to_norm_coordinates(x, y, total_width, total_height):
    """Reformat absolute coordinates of image of shape shape to normalized ones.

    :param int x: x coordinate of pt in px
    :param int y: y coordinate of pt in px
    :param int total_width: total width of the image the point shall be normalized in
    :param int total_height: total height of the image the point shall be normalized in
    :return:
    """
    return x / total_width, y / total_height


def get_center_points(center_point_width, center_point_height, sliding_window_size=(1, 1), col_wise=False):
    """Create list of center points of boxes of size center_point_width x center_point_height.

    :param int center_point_width: box width (normalized)
    :param int center_point_height: box height (normalized)
        of the box that an upscaled center point (pixel) would take up in the image
    :param tuple sliding_window_size: (width, height) of a sliding window in number of center points;
        only center points that can be center of a sliding window that lies completely within the image
        are valid
    :param boolean col_wise: whether to join the columns to a list instead of the rows
    :return: list row-wise (resp. column-wise) center points as (x, y)
    """
    col_border = int(sliding_window_size[0] / 2.)
    row_border = int(sliding_window_size[1] / 2.)
    num_center_pt_cols = int(round(1 / center_point_width)) - 2 * col_border
    num_center_pt_rows = int(round(1 / center_point_height)) - 2 * row_border

    center_pts = np.empty(shape=[num_center_pt_rows, num_center_pt_cols, 2])

    # Create points:
    # Exclude center points whose sliding window would cross the image border
    for row in range(0, num_center_pt_rows):
        for col in range(0, num_center_pt_cols):
            center_pts[row, col, 0] = (col + col_border + 0.5) * center_point_width  # x
            center_pts[row, col, 1] = (row + row_border + 0.5) * center_point_height  # y

    # Concatenate;
    # If col_wise concatenation is required, transpose first
    if col_wise:
        center_pts = list(center_pts.T)
    return np.concatenate(center_pts, axis=0)


def get_anchor_boxes(center_points, anchor_shapes):
    """
    Mind, that the order has to be compatible with the one induced by reshaping
    in the rpn layers.

    :param List[Tuple[float, float]] center_points: list of center points (x, y)
        in normalized coordinates for the anchors
    :param list anchor_shapes: list of anchor shapes as (w, h) in normalized coordinates
    :return: anchor box coordinates as [x1, y1, x2, y2] in normalized coordinates
    as an np.array arranged as
        [... (row i, shape j), (row i, shape j+1), ..., (row i+1, shape j), ...]
    The indices of anchors crossing a border.
    """
    anchors = [[x - w / 2, y - h / 2, x + w / 2, y + h / 2]
               for (x, y) in center_points
               for (w, h) in anchor_shapes
               ]

    border_cross_anchor_indices = []
    for i in range(0, len(anchors)):
        x1, y1, x2, y2 = anchors[i]
        if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
            border_cross_anchor_indices.append(i)

    return np.array(anchors), border_cross_anchor_indices


class Config(DictlikeConfig):
    """Configuration parameters. To adapt, create sub-class and overwrite."""
    CONFIG_STORE_FILEPATH = "config.json"

    ###################
    # TRAINING CONFIG #
    ###################
    BATCH_SIZE = 32  # 64 gets quite slow
    EPOCHS = 20
    BACKBONE_PRETRAINING_BATCH_SIZE = 128
    BACKBONE_PRETRAINING_EPOCHS = 15
    VALIDATION_SPLIT = 0.1

    ###################################
    # DATA GENERATION SPECIFIC CONFIG #
    ###################################
    # Factor by which the number of negative samples may exceed the number of positive samples.
    BALANCE_FACTOR = 5  # TODO: optimize BALANCE_FACTOR

    # Number of times an image is repeatedly read in for training,
    # each time with a different collection of negative classified bounding boxes.
    # All other negative ground-truth bounding boxes get marked as neutral
    # thus not contributing to the rpn_cls_loss, in order to balance the amount
    # of positive and negative samples for rpn_cls.
    NUM_BALANCED_REPETITIONS = 1

    # Image simplification settings
    GRAYSCALE = True
    INVERTED_COLORS = True
    NORMALIZE_IMAGE = True

    ####################
    # DATA PREPARATION #
    ####################
    # input image shape: [height, width, channels]
    # height, width have to be divisible by 2**3!
    IMAGE_SHAPE = [256, 256]
    # Same for the backbone pretraining input
    BACKBONE_TRAINING_IMAGE_SHAPE = [32, 32]

    # Maximum iou value of an anchor with an object's bounding box
    # below which the anchor is considered not to contain the object.
    # 0.3: min. half overlap for similar sized boxes
    # 0.5: min. 2/3 overlap for similar sized boxes
    MAX_IOU_NEGATIVE = 0.5  # Original implementation: 0.3

    # Minimum iou value of an anchor with an object's bounding box
    # above which the anchor is considered to definitely contain the object.
    MIN_IOU_POSITIVE = 0.7

    ##################
    # NETWORK CONFIG #
    ##################
    # What window size does one input pixel for the RPN region
    # and the RPN class network correspond to?
    # BACKBONE_DOWNSCALE * RPN_SHARED_DOWNSCALE
    # !! Must not exceed any of the image dimensions !!
    DOWNSCALING_FACTOR = 2 ** 4

    # (width, height) of one sliding window; should both be odd!
    # in [px/DOWNSCALING_FACTOR] resp. [number of center points]
    SLIDING_WINDOW_SIZE = (3, 3)

    # Number of classification classes (10 for MNIST)
    NUM_BACKBONE_PRETRAINING_CLASSES = 10  # Should actually include "None", thus 11

    # Number of proposals to be processed for masks;
    # !! Should be higher than the the maximum number of objects with bounding boxes per image !!
    NUM_PROPOSALS = 18

    # number of proposals to be processed before NMS,
    # selected by best foreground score
    PRE_NMS_LIMIT = 30

    # Threshold of foreground score for Non-Maximum Suppression:
    # iou above which two boxes are considered to contain the same object
    NMS_THRESHOLD = 0.2  # Original implementation: 0.3

    RPN_CLS_LOSS_NAME = "rpn_cls_loss"
    RPN_REG_LOSS_NAME = "rpn_reg_loss"
    LOSS_LAYER_NAMES = [
        RPN_CLS_LOSS_NAME,
        RPN_REG_LOSS_NAME
    ]

    # Parameters used for SGD optimizer
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    GRADIENT_CLIP_NORM = 5.0

    def __init__(self):
        # Adjust shapes according to simplification:
        # Set number of channels depending on grayscale
        channels = 1 if self.GRAYSCALE else 3
        self.IMAGE_SHAPE.append(channels)
        self.BACKBONE_TRAINING_IMAGE_SHAPE.append(channels)

        # number of different anchor shapes per center
        self.ANCHOR_SHAPES = [
            to_norm_coordinates(29, 30, total_width=self.IMAGE_SHAPE[1], total_height=self.IMAGE_SHAPE[0]),
            to_norm_coordinates(15, 33, total_width=self.IMAGE_SHAPE[1], total_height=self.IMAGE_SHAPE[0])
        ]
        self.NUM_ANCHOR_SHAPES = len(self.ANCHOR_SHAPES)

        # Get center points (normalized coordinates)
        # A center point corresponds to a quadratic box of width DOWNSCALING_FACTOR.
        # From this calculate the normalized center point coordinates.
        center_pt_width = self.DOWNSCALING_FACTOR / self.IMAGE_SHAPE[1]
        center_pt_height = self.DOWNSCALING_FACTOR / self.IMAGE_SHAPE[0]
        # list of (x, y) in normalized coordinates
        self.CENTER_POINTS = get_center_points(center_point_height=center_pt_height,
                                               center_point_width=center_pt_width,
                                               sliding_window_size=self.SLIDING_WINDOW_SIZE)

        # Get anchor box coordinates (normalized coordinates)
        # and indices of anchors crossing a border.
        self.ANCHOR_BOXES, self.BOUNDARY_ANCHOR_INDICES = get_anchor_boxes(self.CENTER_POINTS, self.ANCHOR_SHAPES)
        self.NUM_ANCHORS = self.ANCHOR_BOXES.shape[0]

        # Loss weights (compare original implementation and corresponding paper)
        self.LOSS_WEIGHTS = {
            self.RPN_CLS_LOSS_NAME: 1,
            self.RPN_REG_LOSS_NAME: 1
        }

        # Input shapes for the RPN network: objectness class and region coordinates
        self.RPN_CLS_SHAPE = [self.NUM_ANCHORS, 1]
        self.RPN_REG_SHAPE = [self.NUM_ANCHORS, 4]

        # VALIDITY CHECKS
        self.validity_checks()

    def validity_checks(self):
        """Check validity of given entries for prototyping."""
        assert self.DOWNSCALING_FACTOR <= self.IMAGE_SHAPE[0] and self.DOWNSCALING_FACTOR <= self.IMAGE_SHAPE[1]
