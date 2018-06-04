#! /bin/python3
"""
Generate and load simple, labeled image segmentation data
based on MNIST(-like) single object images.

Functionalities:

* Crop and mask images:
  Detect the single object in simple images (like MNIST letter files),
  crop the images to a bounding box of this object, and produce a mask.
  Cropped versions and masked versions are then stored under the original
  name in different root directories using the same folder architecture,
  which at the same time represents the labels.
  Main Function: crop_and_mask_mnist()
* Load data like the mnist-module
  To work with the cropped image versions instead of the originals,
  there is a load_data() function analogues to the module keras.datasets.mnist.
  It also ensures the cropped versions are generated before loading.
  Main Function: load_data()
* Generate random images with labels:
  There are tools to create batches of images, each containing a random amount
  of the cropped objects from above, and a random amount of further layers (like stains)
  taken from other sample images. For the numerous possible settings have a look at the
  GenerationConfig class, which is used for settings.
  The images and corresponding annotations are saved in batches, where the annotations
  are in a COCO-like format.
  Main Function: generate_labeled_data_files()
* Load generated data:
  Load the previously stored data into numpy arrays.
  Main Function: load_labeled_data()

All main functions accept a GenerationConfig configuration object for generation specific
settings (like storage locations or randomization parameters).
"""

from config import GenerationConfig
import os
import random

import numpy as np
import cv2
import json
from tqdm import tqdm


# CV2 DIMENSION CONVENTIONS:
# cv2.resize(img, (w, h))
# img.shape = (h, w, channels)
# img[h:w]

# --------------------
# HELPER FUNCTIONS
# --------------------
def otherroot(imagefileroot, new_mnist_root, image_mnist_root):
    """Returns (and creates) path with exchanged root folder for given image file."""
    newimagefileroot = imagefileroot.replace(image_mnist_root, new_mnist_root, 1)
    if not os.path.isdir(newimagefileroot):
        os.makedirs(newimagefileroot)
    return newimagefileroot


def write_image(imagefile, image):
    cv2.imwrite(imagefile, image)


def load_image_with_resolution(imagefile, image_shape=None):
    """Load image from file, reshape to image_shape and return original shape.

    :param str imagefile: full path to the image to read in
    :param tuple image_shape: shape-like tuple of at least (height, width)
    :return: image as np.array, original_shape = [height, width, channels]
    """
    image = cv2.imread(imagefile)
    original_shape = image.shape
    # Resize image if it is not of required resolution
    if image_shape is not None and \
            (image.shape[0] != image_shape[0] or image.shape[1] != image_shape[1]):
        image = cv2.resize(image, (image_shape[1], image_shape[0]))
    return image, original_shape


def load_image(imagefile):
    """Load image from file.

    :param str imagefile: full path to the image to read in
    :return: image as np.array
    """
    return cv2.imread(imagefile)


def resized_image_from_file(root, imagefilename,
                            avg_height=None, max_height_variance=None,
                            fixed_scaling_factor=None):
    """Get image from file and resize by fixed_scaling_factor or randomly.

    :param str root: root of imagefile
    :param str imagefilename: filename
    :param float fixed_scaling_factor: rescale image by this factor;
            either this or avg_height, max_size_variance have to be set
    :param int avg_height: if fixed_scaling_factor is not set,
            the image is rescaled to have a height within
            [avg_height-max_height_variance, avg_height+max_height_variance]
    :param float max_height_variance: see avg_height
    """
    # obtain cv2 images
    imagefile = os.path.join(root, imagefilename)
    image = load_image(imagefile)

    # resize
    if fixed_scaling_factor is None:
        h = image.shape[0]  # current letter height
        dh = (random.random() * 2 - 1) * max_height_variance * avg_height  # random difference from avg_height
        fixed_scaling_factor = (avg_height + dh) / h
    image = cv2.resize(image, None, fx=fixed_scaling_factor, fy=fixed_scaling_factor)

    return image, fixed_scaling_factor


def simple_resize(image, resolution):
    """Resize image to dimension = (width, height).

    :param np.array image: image to resize
    :param tuple resolution: (width, height)
    """
    return cv2.resize(image, resolution)


def invert(image):
    """Invert image colors."""
    return cv2.bitwise_not(image)


def to_grayscale(image):
    """Invert BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_bgr_colorspace(image):
    """Invert BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def mask_by_threshold(image, thresh, inverted):
    """Obtain mask from image by grayscaling and thresholding.
   :param image: cv2 image to threshold
   :param int thresh: threshold
   :param bool inverted: whether the image is already inverted"""
    if not inverted:
        image = invert(image)
    # change to grayscale colorspace
    image_grayscale = to_grayscale(image)
    return_code, image_thresholded = cv2.threshold(
        src=image_grayscale,
        thresh=thresh,
        maxval=255,
        type=0  # cv2.THRESH_BINARY ?
    )
    return image_thresholded


def max_bounding_box(mask):
    """Get corner points of joint bounding box of mask as (xmin, ymin), (xmax, ymax).
   :param mask: mask to search for contour bounding boxes.
   """
    # get contours
    mask, contours, hierarchy = cv2.findContours(
        image=mask,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE  # as few points to describe mask as possible
    )
    # collect bounding box rectangles of contours
    boxes_xmin, boxes_ymin, wstart, hstart = cv2.boundingRect(contours[0])
    boxes_xmax, boxes_ymax = boxes_xmin + wstart, boxes_ymin + hstart
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes_xmin, boxes_ymin = min(x, boxes_xmin), min(y, boxes_ymin)
        boxes_xmax, boxes_ymax = max(x + w, boxes_xmax), max(y + h, boxes_ymax)
    return (boxes_xmin, boxes_ymin), (boxes_xmax, boxes_ymax)


def do_boxes_intersect(box, *other_boxes):
    """Whether the first box intersects with any of the others.

    :param tuple box: box described by coordinates of upper left corner
        and width, height by ((x,y),(w,h)) with x, y, w, h integers
    :param list other_boxes: list of other boxes of the form ((x,y), (width,height))
    :return: bool
    """
    ((x, y), (w, h)) = box
    for ((other_x, other_y), (other_w, other_h)) in other_boxes:
        # pt lies in box?
        if ((other_x <= x <= other_x + other_w) or (x <= other_x <= x + w)) and \
                ((other_y <= y <= other_y + other_h) or (y <= other_y <= y + h)):
            return True
    return False


def crop_and_mask(image, thresh, inverted):
    """Crop image to largest bounding box.

    :param str image: cv2 image to get a cropped version from
    :param int thresh: threshold for obtaining mask; see mask_by_threshold()
    :param bool inverted: whether the image is already inverted
       (MNIST: yes, i.e. letters white, background black)
   """
    # obtain binary mask for letter (letter white)
    mask = mask_by_threshold(image, thresh, inverted=inverted)

    # crop to bounding box
    box_pt1, box_pt2 = max_bounding_box(mask)
    xmin, ymin = box_pt1
    xmax, ymax = box_pt2
    image = image[ymin:ymax, xmin:xmax]
    mask = mask[ymin:ymax, xmin:xmax]

    return image, mask


def generate_crop_and_mask_files(imagefile, maskoutfile, cropoutfile, thresh, inverted=True):
    """Load an image from file, crop_and_mask it, store mask and cropped output.

    :param inverted:
    :param str imagefile: path to image to be loaded
    :param str maskoutfile: path to mask output file; overwrites if exists
    :param str cropoutfile: path to crop output file; overwrites if exists
    :param int thresh: threshold for obtaining mask; see crop_and_mask()
    """
    # load image (MNIST: letters white, background black)
    image = load_image(imagefile)
    crop, mask = crop_and_mask(image, thresh=thresh, inverted=inverted)
    # output
    write_image(maskoutfile, mask)
    write_image(cropoutfile, crop)


def crop_and_mask_mnist(config):
    """Mask and crop all images in config.MNIST_PNG_ROOT and save to
   config.MNIST_MASK_ROOT, config.MNIST_CROP_ROOT.

   :param GenerationConfig config: configuration object with fields

   * MNIST_PNG_ROOT, MNIST_MASK_ROOT, MNIST_CROP_ROOT (root folders)
   * THRESHOLD (threshold for obtaining mask)
   """
    for root, dirs, files in os.walk(config.MNIST_PNG_ROOT):
        # create folders
        print("Processing folder", root, "...")
        maskroot = otherroot(root, config.MNIST_MASK_ROOT, config.MNIST_PNG_ROOT)
        croproot = otherroot(root, config.MNIST_CROP_ROOT, config.MNIST_PNG_ROOT)
        if not os.path.isdir(croproot):
            os.makedirs(croproot)

        # process files
        for filename in files:
            maskfile = os.path.join(maskroot, filename)
            cropfile = os.path.join(croproot, filename)
            imagefile = os.path.join(root, filename)
            generate_crop_and_mask_files(imagefile=imagefile, maskoutfile=maskfile, cropoutfile=cropfile,
                                         thresh=config.THRESHOLD)


def load_labeled_data_from_folder(img_folder_name,
                                  img_folder_root,
                                  mask_folder_root,
                                  image_resolution,
                                  resizefunc=simple_resize,
                                  convert_to_gray=True):
    """Produce lists of (imgs, labels, masks), all images as np.array, from folder hierarchy.

    Needs a folder hierarchy of

    |-folderroot
        |-foldername
            |-folders with labels as name (e.g. 1, 2, 3 ...)
               |-images with that label

    :param str img_folder_name: folder name where the images lie within under the label-folders
    :param str img_folder_root: path to the folder
    :param str mask_folder_root: path to the folder
    :param tuple image_resolution: dimension of the output images
    :param func resizefunc: function for resizing of the form (image, resolution) -> resized_image
    :param boolean convert_to_gray: whether to convert the image to grayscale
    :return: tuple (xs, labels, masks) of lists:
        - xs: images resized to imagedim
        - labels: labels (int) for images
        - masks: masks resized to imagedim
    """
    labeled_data = []
    folderpath = os.path.join(img_folder_root, img_folder_name)
    for path, dirs, files in os.walk(folderpath):
        label = os.path.basename(path)
        print("Processing label", label, "in folder", folderpath)
        for filename in files:
            # x
            imagefile = os.path.join(path, filename)
            image = load_image(imagefile)
            image = resizefunc(image, image_resolution)
            if convert_to_gray:
                image = to_grayscale(image)  # 1 channel

            # mask
            maskfileroot = otherroot(path, mask_folder_root, img_folder_root)
            maskfile = os.path.join(maskfileroot, filename)
            mask = load_image(maskfile)
            mask = resizefunc(mask, image_resolution)
            if convert_to_gray:
                mask = to_grayscale(mask)

            labeled_data.append((image, label, mask))
    x, label, masks = tuple(zip(*labeled_data))
    return np.array(x), np.array(label), np.array(masks)


def load_data(config=GenerationConfig(),
              do_convert_to_gray=True,
              do_resize=True):
    """Load MNIST training and test data as (cropped img, mask, label).

   Needsdirectory structure

   |-root images
   |   |-label folders
   |      |-images
   |-root mask
   |   |-label folders
   |      |-images
   |-root crop
      ...

    :param GenerationConfig config: config object with fields

        * MNIST_CROP_ROOT and MNIST_MASK_ROOT
          (the root folders of cropped images resp. masks as .jpg or .png)
        * TRAIN_FOLDER, TEST_FOLDER (folder names of train and test data)
        * LETTER_RESOLUTION (output image dimension)

    :param boolean do_resize: whether to apply the default resize function or not
    :param boolean do_convert_to_gray: whether to convert images to grayscale
    :return: (train, test), where each is a list of tuples (xs, labels, masks)
        as returned by load_labeled_data_from_folder()
   """
    print("Loading data ...")

    # possible preparation
    if not os.path.isdir(config.MNIST_CROP_ROOT) or \
            not os.path.isdir(config.MNIST_MASK_ROOT):
        crop_and_mask_mnist(config)

    # iterate over labels
    further_args = {}
    if not do_resize:
        further_args = {"resizefunc": lambda x: x}
    return tuple([
        load_labeled_data_from_folder(
            folder, config.MNIST_CROP_ROOT, config.MNIST_MASK_ROOT,
            convert_to_gray=do_convert_to_gray,
            image_resolution=config.LETTER_RESOLUTION,
            **further_args)
        for folder in [config.TRAIN_FOLDER, config.TEST_FOLDER]
    ])


# ------------
# RANDOM UTILS
# ------------
def random_letter(labelroot):
    """Randomly pick an image from a random label folder.

    Needs a folder structure of:
    labelroot
    |--label folders
       |--images

    :param str labelroot: path to the directory, where the label folders lie in

    :return: label, imgroot, randimgfile:
    label, root, and filename of the randomly chosen image
    """
    randlabel = random.choice(
        [label for label in os.listdir(labelroot) if os.path.isdir(os.path.join(labelroot, label))]
    )
    imgroot = os.path.join(labelroot, randlabel)
    randimgfile = random_file(imgroot)
    return randlabel, imgroot, randimgfile


def random_file(root):
    return random.choice(
        [img for img in os.listdir(root) if os.path.isfile(os.path.join(root, img))]
    )


def random_color(max_brightness, valid_colors=None):
    """Randomly pick a color with a brightness at most max_brightness.

    :param int max_brightness: maximum value of the sum of all channels;
    has to be in [0, 3*255]
    :param list valid_colors: valid colors of brightness at most max_brightness;
    will be set to the GenerationConfig default for this max_brightness if None
    """
    if valid_colors is None:
        valid_colors = GenerationConfig.valid_colors(max_brightness)
    randcolor = random.choice(valid_colors)
    return randcolor


def random_window(image, window_xdim, window_ydim):
    """Get a random window of specified dimensions from image.

    :param np.array image: image as np.array
    :param int window_xdim: width of the window in px
    :param int window_ydim: height of the window in px
    :return: random window as np.array taken from image;
        None if no valid position is available
    """
    img_ydim, img_xdim = image.shape[0], image.shape[1]
    window_aspect_ratio = window_ydim / window_xdim

    # max window dimensions with above aspect ratio
    window_maxw = min(img_xdim, int(img_ydim / window_aspect_ratio))
    # random window width, height within above bounds
    w = random.randint(window_xdim, window_maxw) if window_maxw > window_xdim else window_maxw
    h = int(window_aspect_ratio * w)
    # random position of window in texture image
    pt = random_anchor(w, h, img_xdim, img_ydim)
    # If the window did not fit into the image, return the whole image
    if pt is None:
        return cv2.resize(image, (window_xdim, window_ydim))
    x, y = pt

    # extract window as image and resize to wanted dimensions
    window = image[y:y + h, x:x + w]
    window = cv2.resize(window, (window_xdim, window_ydim))
    return window


def random_coord(xmin, xmax, ymin, ymax):
    return random.randint(xmin, xmax), random.randint(ymin, ymax)


def random_anchor(w, h, image_width, image_height,
                  occupied_boxes=None,
                  valid_box_anchors=None):
    """
    :param int w: width of the window in px
    :param int h: height of the window in px
    :param int image_width: width of the complete image
    :param int image_height: height of the complete image
    :param list occupied_boxes: list/tuple of boxes of the form ((x,y), (width, height))
        the window may not intersect with
    :param valid_box_anchors: set of box anchors to randomly choose one from;
        default: every point such that a box having its upper left corner at this point,
        and width w, and height h does not intersect any occupied box
    :return: (x,y) valid random coordinate; None if no valid position is available
    """
    if occupied_boxes is None:
        return random.randint(0, image_width - w), random.randint(0, image_height - h)
    valid_box_anchors = valid_box_anchors or [
        (x, y)
        for x in range(0, image_width - w)
        for y in range(0, image_height - h)
        if not do_boxes_intersect(((x, y), (w, h)), *occupied_boxes)
    ]
    if len(valid_box_anchors) == 0:
        return None
    return random.choice(valid_box_anchors)


def apply_layer_from_random_file(image, layerroot, weight1=0.9, weight2=0.4):
    """Apply an image from file as layer to the given image.

    :param nd.array image: image data
    :param str layerroot: path to root folder of layer images
    :param float weight1: weight of image for blending
    :param float weight2: weight of layer for blending
    :return: image data with applied layer
    """
    ydim, xdim, channels = image.shape
    layerfile = random_file(layerroot)
    layer = invert(load_image(os.path.join(layerroot, layerfile)))
    image = cv2.addWeighted(image, weight1,
                            random_window(layer, xdim, ydim), -weight2,
                            0)
    return image


# ----------------
# MATCH FORMAT TRANSLATIONS
# ----------------
def to_match_dict(config, label, bounding_box, mask):
    """Return properly formatted dict that can be extracted by match_to_tuple.

    :param GenerationConfig config: configuration object with keys
    JSON_LABEL_KEY, JSON_BOUNDING_BOX_KEY, JSON_MASK_KEY
    :param str label: label
    :param tuple bounding_box: bounding box tuple as ((x1,y1), (x2,y2))
    :param np.array mask: mask image
    :return: properly formatted dict"""
    return {
        config.JSON_LABEL_KEY: label,
        config.JSON_BOUNDING_BOX_KEY: bounding_box,
        config.JSON_MASK_KEY: mask.tolist()
    }


def match_to_tuple(config, match, mask_resolution=None):
    """Reformat dict match to tuple.

    :param GenerationConfig config: configuration object containing needed keys
    :param dict match: match with keys JSON_LABEL_KEY, JSON_BOUNDING_BOX_KEY, JSON_MASK_KEY
        and values as specified for these keys
    :param tuple mask_resolution: resolution the mask is resized to as (width, height);
    not applied if None
    :return: tuple (label, bounding box, mask in mask_resolution)
    """
    mask = np.array(match[config.JSON_MASK_KEY])
    if mask_resolution is not None:
        mask = cv2.resize(mask, mask_resolution)

    # ensure integer coordinates
    # see https://stackoverflow.com/a/43656642
    (x1, y1), (x2, y2) = match[config.JSON_BOUNDING_BOX_KEY]
    bounding_box = ((int(x1), int(y1)), (int(x2), int(y2)))

    return (
        match[config.JSON_LABEL_KEY],
        bounding_box,
        mask
    )


# -----------------------------
# Image generation
# -----------------------------
def generate_random_image(config):
    """Randomly generate an image containing letters.

    :param GenerationConfig config: config object with fields

    * MIN_NUM_LETTERS, MAX_NUM_LETTERS
    * MIN_LETTERHEIGHT, MAX_LETTERHEIGHT
    * MAX_OVERLAP
    * MAX_HEIGHT_VARIANCE
    * IMAGE_RESOLUTION
    * MAX_BRIGHTNESS
    * MAX_ALPHA
    * DEBUG

    :return: tuple (np.array image, list matches) where
        image is a randomly created cv2 image,
        matches is a list of dicts of the form
            {
                JSON_BOUNDING_BOX_KEY: box coordinates as ((x1, y2), (x2, y2)),
                JSON_LABEL_KEY: label,
                JSON_MATCHES_KEY: mask converted from cv2 image to nested list
            }
    """
    matches = []
    image_width, image_height = config.IMAGE_RESOLUTION[0], config.IMAGE_RESOLUTION[1]

    # Start with empty white image
    image = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 255

    # TODO: Generalize to arbitrary number of overlays
    # TEXTURE
    if config.TEXTURES_FOLDER is not None:
        texturefile = random_file(config.TEXTURES_FOLDER)
        # In case of Errors of the form "Premature end of JPEG file",
        # ensure the images have been downloaded properly (and no .crdownload-files are processed)
        texture = load_image(os.path.join(config.TEXTURES_FOLDER, texturefile))
        # (randomly) extract window for image
        image = random_window(texture, image_width, image_height)

    # GRID
    if config.GRIDS_FOLDER is not None:
        image = apply_layer_from_random_file(image, layerroot=config.GRIDS_FOLDER)

    # (STAINS)
    if config.STAINS_FOLDER is not None:
        image = apply_layer_from_random_file(image, layerroot=config.STAINS_FOLDER)

    # LETTERS
    num_letters = random.randint(config.MIN_NUM_LETTERS, config.MAX_NUM_LETTERS)
    if num_letters == 0:
        return image, ()

    letterfiles = list(map(
        lambda r: random_letter(os.path.join(config.MNIST_CROP_ROOT, config.TRAIN_FOLDER)),
        range(0, num_letters)
    ))

    # COMMON PARAMETERS
    # Average height; maximum depending on number of letters
    max_letterheight = min(config.MAX_LETTERHEIGHT - int(round(config.MAX_LETTERSIZE_DECREASE_FACTOR * num_letters)),
                           config.MIN_LETTERHEIGHT + 1)
    avg_height = random.randint(config.MIN_LETTERHEIGHT, max_letterheight)
    # weight for mixing in the letter
    letter_weight = 1 if config.MAX_ALPHA == 0 else 1 - (1 - random.random()) * config.MAX_ALPHA
    if config.DEBUG:
        print("Average height:\t", avg_height)
        print("Alpha factor:\t", letter_weight)

    # loop over chosen letters
    occupied_boxes = []
    for label, root, l in letterfiles:
        letter, scale = resized_image_from_file(root, l,
                                                avg_height=avg_height,
                                                max_height_variance=config.MAX_HEIGHT_VARIANCE)
        mask, scale = resized_image_from_file(otherroot(root, config.MNIST_MASK_ROOT, config.MNIST_CROP_ROOT), l,
                                              fixed_scaling_factor=scale)

        # COLOR: apply random color factors to letter channels
        letter = randomly_color_inverted_image(letter, config.MAX_BRIGHTNESS)

        # coordinates
        w, h = letter.shape[1], letter.shape[0]
        pt = random_anchor(w, h, image_width, image_height, occupied_boxes=occupied_boxes)
        # If no space was left on image, try with next letter
        if pt is None:
            continue
        x, y = pt

        # apply letter to image
        image[y:y + h, x:x + w] = cv2.addWeighted(image[y:y + h, x:x + w], 1, letter, -letter_weight, 0)

        # add letter mask, label, and coord. to matches
        occupied_boxes.append(((x + config.MAX_OVERLAP, y + config.MAX_OVERLAP),
                               (w - 2 * config.MAX_OVERLAP, h - 2 * config.MAX_OVERLAP)))
        matches.append(to_match_dict(config,
                                     label=label,
                                     bounding_box=((x, y), (x + w, y + h)),
                                     mask=mask))

    # if config.DEBUG:
    #     # draw metadata into image
    #     image = draw_masks_and_labels(image,
    #                                   map(lambda m: match_to_tuple(config, m), matches))

    return image, matches


def randomly_color_inverted_image(image, max_brightness):
    """Applies color of max_brightness to inverted, 3-channel grayscale image.

    :param image: 3-channel grayscale image to apply color to
    :param max_brightness: in [0, 3*255]; maximum sum of all channels of a pixel
    in the non-inverted version of the output image
    """
    if max_brightness is None or max_brightness == 0:  # Nothing to do
        return image
    channel_factors = tuple(map(lambda col: (255 - col) / 255,
                                random_color(max_brightness)))
    for i in range(0, image.shape[2]):
        image[:, :, i] = np.multiply(image[:, :, 0], channel_factors[i])
    return image


def generate_labeled_data_files(config):
    """Generate num_samples images with annotations and save them.

    The annotations are saved in a json file of the format
    [
        {
            JSON_FILEPATH_KEY: path to the image file (cv2-readable format) relative to call position
            JSON_MATCHES_KEY: [
                # list of matches with format as in create_random_image
            ]
        }
    ]

    :param GenerationConfig config: configuration object
    """

    #  ensure folders exist
    if not os.path.isdir(config.DATA_IMAGEROOT):
        os.makedirs(config.DATA_IMAGEROOT)
    if not os.path.isdir(config.DATA_ANNOTATIONSROOT):
        os.makedirs(config.DATA_ANNOTATIONSROOT)

    print("Generation configuration:\n" + str(config))
    config.to_json_file()

    start_id_enumeration = config.START_ID_ENUMERATION
    for batch_id in tqdm(range(0, config.NUM_BATCHES), "Image batches"):
        annotations = []
        for img_id in tqdm(range(start_id_enumeration, start_id_enumeration + config.BATCH_SIZE),
                           "Images for batch " + str(batch_id)):
            image, matches = generate_random_image(config)
            imagefilename = config.IMAGE_FILENAME_FORMAT.format(img_id)
            imagefile = os.path.join(config.DATA_IMAGEROOT, imagefilename)

            # save image
            write_image(imagefile, image)
            # note annotation
            annotations.append({
                config.JSON_FILENAME_KEY: imagefilename,
                config.JSON_MATCHES_KEY: matches
            })

        # save all annotations
        annotations_filename = config.ANNOTATIONS_FILENAME_FORMAT.format(batch_id)
        annotationsfile = os.path.join(config.DATA_ANNOTATIONSROOT, annotations_filename)
        with open(annotationsfile, 'w+') as annotations_filehandle:
            json.dump(annotations, annotations_filehandle)

        start_id_enumeration += config.BATCH_SIZE


# -----------
# LOAD TOOLS
# -----------

def load_labeled_data(config, image_shape=None, mask_resolution=None):
    """Read in images and annotations as specified in annotationsfiles found in annotationsroot.

    :param GenerationConfig config: configuration object with fields

    * DATA_ANNOTATIONSROOT, DATA_IMAGEROOT, JSON_MASK_RESOLUTION
    * JSON_FILENAME_KEY, JSON_MATCHES_KEY
    * JSON_LABEL_KEY, JSON_BOUNDING_BOX_KEY, JSON_MASK_KEY

    :param image_shape: shape in list form [height, width, ...] to resize loaded images to
    :param tuple mask_resolution: resolution as (width, height) to which the masks are resized;
    original shape is kept if set to None
    :return: tuple of lists of the format
        (list of images as np.arrays,
        list of lists of matches,
        list of original image shapes as [height, width, channels])
        where the matches are tuples of the form specified in match_to_tuple()
    """
    # TODO: make a generator read in only batches of certain size
    data = []
    for root, dirs, annotationsfiles in os.walk(config.DATA_ANNOTATIONSROOT):
        for annotationsfilename in tqdm(annotationsfiles, "Annontationfiles", leave=False):
            annotationsfile = os.path.join(root, annotationsfilename)
            with open(annotationsfile, 'r') as annotations_filehandle:
                annotations = json.load(annotations_filehandle)

            for annotation in tqdm(annotations, "Image files", leave=False):
                # image
                imagefile = os.path.join(config.DATA_IMAGEROOT, annotation[config.JSON_FILENAME_KEY])
                image, original_shape = load_image_with_resolution(imagefile, image_shape=image_shape)

                matches = list(map(
                    lambda match: match_to_tuple(config, match, mask_resolution),
                    annotation[config.JSON_MATCHES_KEY]))
                data.append((image, matches, original_shape))

    return list(zip(*data))


# -------------------
# INSPECTION TOOLS
# -------------------

def draw_bounding_boxes(image, boxes, color, thickness=1):
    """Return the image with bounding boxes.

    :param np.array image: image
    :param iterator boxes: list of bounding boxes as ((x1, y1), (x2, y2))
    :param tuple color: rgb color tuple
    :param int thickness: line thickness of the bounding box in px
    :return: image with bounding boxes
    """
    for box in boxes:
        cv2.rectangle(image,
                      box[0],  # (x1, y1)
                      box[1],  # (x2, y2)
                      color,
                      thickness=thickness)
    return image


def draw_masks(image, matches, color):
    bounding_boxes = list(map(lambda m: m[1], matches))
    image = draw_bounding_boxes(image, bounding_boxes, color)
    for match in matches:
        (x1, y1), (x2, y2) = match[1][0], match[1][1]
        mask = simple_resize(match[2], (x2 - x1, y2 - y1)).astype(image.dtype)
        image[y1:y2, x1:x2] = cv2.addWeighted(image[y1:y2, x1:x2], 1,
                                              mask, -1,
                                              0)
    return image


def draw_masks_and_labels(image, matches, color=(0, 255, 0)):
    image = draw_masks(image, matches, color)
    font, font_scale, thickness = cv2.FONT_HERSHEY_PLAIN, 1, 1
    for match in matches:
        label = match[0]

        # background of label
        text_width, text_height = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x1, y1 = match[1][0]
        bottom_left_corner = (x1, y1 + text_height)
        image[y1:y1 + text_height, x1:x1 + text_width] = 0

        # label
        image = cv2.putText(image,
                            label,
                            bottom_left_corner,
                            fontFace=font,
                            fontScale=font_scale,
                            color=color,
                            thickness=1)
    return image


if __name__ == "__main__":
    # Generate cropped letter versions and masks
    load_data()

    # Generate labeled image data
    generate_labeled_data_files(GenerationConfig().update({
        "BATCH_SIZE": 50,
        "NUM_BATCHES": 110
    }))
