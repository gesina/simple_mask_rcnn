#! /bin/python3
import os
import random

import numpy as np
import cv2
import json

# CV2 DIMENSION CONVENTIONS:
# cv2.resize(img, (w, h))
# img.shape = (h, w, channels)
# img[h:w]

# CROP and MASK GENERATION
THRESHOLD = 100
BOUNDING_BOX_COLOR = (0, 255, 0)
MAX_BOUNDING_BOX_COLOR = (0, 0, 255)
MNIST_PNG_FOLDER = "data/mnist_png"
MNIST_CROP_ROOT = "data/mnist_crop"
MNIST_MASK_ROOT = "data/mnist_mask"
LETTER_RESOLUTION = (28, 28)

# IMAGE GENERATION
TEXTURES_FOLDER = "data/textures"
GRIDS_FOLDER = "data/grids"
STAINS_FOLDER = "data/stains"

def valid_colors(max_brightness):
    return tuple(((r, g, b) for r in range(0, 255)
                  for g in range(0, 255)
                  for b in range(0, 255)
                  if r + g + b < max_brightness))
MAX_BRIGHTNESS = 200
VALID_COLORS = {
    MAX_BRIGHTNESS: valid_colors(MAX_BRIGHTNESS)
}

# JSON
JSON_MASK_KEY = "match"
JSON_MATCHES_KEY = "matches"
JSON_LABEL_KEY = "label"
JSON_BOUNDING_BOX_KEY = "bounding_box"
JSON_FILENAME_KEY = "filepath"

JSON_MASK_RESOLUTION = (14, 14) # compare letter resolution
DATA_IMAGEROOT = "data/mask_rcnn/images"



def otherroot(imagefileroot, new_mnist_root, image_mnist_root=MNIST_PNG_FOLDER):
    """Returns (and creates) path with exchanged root folder for given image file."""
    newimagefileroot = imagefileroot.replace(image_mnist_root, new_mnist_root, 1)
    if not os.path.isdir(newimagefileroot):
        os.makedirs(newimagefileroot)
    return newimagefileroot


def write_image(imagefile, image):
    cv2.imwrite(imagefile, image)


def load_image(imagefile):
    return cv2.imread(imagefile)


def resized_image_from_file(root, imagefilename,
                            avg_height=None, max_height_variance=None,
                            fixed_scaling_factor=None):
    """Get image from file and resize by fixed_scaling_factor or randomly.

    Parameters:
        root: root of imagefile
        imagefilename: filename
        fixed_scaling_factor: rescale image by this factor;
            either this or avg_height, max_size_variance have to be set
        avg_height, max_height_variance: if fixed_scaling_factor is not set,
            the image is rescaled to have a height within
            [avg_height-max_height_variance, avg_height+max_height_variance]
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


# TODO: RoI-Align
def simple_resize(image, dimension):
    return cv2.resize(image, dimension)


def invert(image):
    """Invert image colors."""
    return cv2.bitwise_not(image)


def mask_by_threshold(image, thresh=THRESHOLD, inverted=True):
    """Obtain mask from image by grayscaling and thresholding.
   :param image: cv2 image to threshold
   :param int thresh: threshold
   :param bool inverted: whether the image is already inverted"""
    if not inverted:
        image = invert(image)
    # change to grayscale colorspace
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return_code, image_thresholded = cv2.threshold(
        src=image_grayscale,
        thresh=thresh,
        maxval=255,
        type=0  # cv2.THRESH_BINARY ?
    )
    return image_thresholded


def max_bounding_box(mask,
                     apply_boxes_image=None,
                     bounding_box_color=BOUNDING_BOX_COLOR):
    """Get corner points of joint bounding box of mask as (xmin, ymin), (xmax, ymax).
   :param mask: mask to search for contour bounding boxes.
   :param apply_boxes_image: The image to draw the intermediate bounding boxes into.
   :param bounding_box_color: see default BOUNDING_BOX_COLOR
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
        if apply_boxes_image is not None:
            cv2.rectangle(apply_boxes_image, (x, y), (x + w, y + h), bounding_box_color)
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
        if ((other_x <= x <= other_x + other_w) or (x <= other_x <= x+w)) and \
           ((other_y <= y <= other_y + other_h) or (y <= other_y <= y+h)):
            return True
    return False


def crop_and_mask(image, inverted=True, debug=False):
    """Crop image to largest bounding box; for debugging only mark bounding boxes.
    
   :param str image: cv2 image to get a cropped version from
   :param bool inverted: whether the image is already inverted
      (MNIST: yes, i.e. letters white, background black)
   :param bool debug: if true, the output is the input with bounding boxes marked.
      Colors are as set in BOUNDING_BOX_COLOR, MAX_BOUNDING_BOX_COLOR.
   """
    # obtain binary mask for letter (letter white)
    mask = mask_by_threshold(image, inverted=inverted)

    # final bounding box
    box_pt1, box_pt2 = max_bounding_box(mask,
                                        apply_boxes_image=image if debug else None)
    if debug:
        cv2.rectangle(image, box_pt1, box_pt2, MAX_BOUNDING_BOX_COLOR)
    else:
        xmin, ymin = box_pt1
        xmax, ymax = box_pt2
        image = image[ymin:ymax, xmin:xmax]
        mask = mask[ymin:ymax, xmin:xmax]

    return image, mask


def generate_crop_and_mask_files(imagefile, maskoutfile, cropoutfile):
    # load image (MNIST: letters white, background black)
    image = load_image(imagefile)
    crop, mask = crop_and_mask(image, debug=False)
    # output
    write_image(maskoutfile, mask)
    write_image(cropoutfile, crop)


def crop_and_mask_mnist(mnist_src_root=MNIST_PNG_FOLDER,
                        mnist_mask_root=MNIST_MASK_ROOT,
                        mnist_crop_root=MNIST_CROP_ROOT):
    """Mask and crop all images in mnist_src_root and save to
   mnist_mask_root, mnist_crop_root."""
    for root, dirs, files in os.walk(mnist_src_root):
        # create folders
        print("Processing folder", root, "...")
        maskroot = otherroot(root, mnist_mask_root, mnist_src_root)
        croproot = otherroot(root, mnist_crop_root, mnist_src_root)
        if not os.path.isdir(croproot):
            os.makedirs(croproot)

        # process files
        for filename in files:
            maskfile = os.path.join(maskroot, filename)
            cropfile = os.path.join(croproot, filename)
            imagefile = os.path.join(root, filename)
            generate_crop_and_mask_files(
                imagefile=imagefile,
                maskoutfile=maskfile,
                cropoutfile=cropfile
            )


def load_labeled_data_from_folder(foldername, folderroot, imagedim=LETTER_RESOLUTION, resizefunc=simple_resize):
    """Produce lists of (imgs, labels, masks), all images as np.array, from folder hierarchy.

    Needs a folder hierarchy of
    |-folderroot
        |-foldername
            |-folders with labels as name (e.g. 1, 2, 3 ...)
               |-images with that label

    :param str foldername: folder name where the images lie within under the label-folders
    :param str folderroot: path to the folder
    :param tuple imagedim: dimension of the output images
    :param func resizefunc: function for resizing of the form (image, imagedim) -> resized_image
    """
    labeled_data = []
    folderpath = os.path.join(folderroot, foldername)
    for path, dirs, files in os.walk(folderpath):
        label = os.path.basename(path)
        print("Processing label", label, "in folder", folderpath)
        for file in files:
            # x
            imagefile = os.path.join(path, file)
            image = load_image(imagefile)
            image = resizefunc(image, imagedim)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 1 channel

            # mask
            maskfileroot = otherroot(path, new_mnist_root=MNIST_MASK_ROOT, image_mnist_root=MNIST_CROP_ROOT)
            maskfile = os.path.join(maskfileroot, file)
            mask = load_image(maskfile)
            mask = resizefunc(mask, imagedim)

            labeled_data.append((image, label, mask))
    x, label, masks = tuple(zip(*labeled_data))
    return (np.array(x), np.array(label), np.array(masks))


def load_data(
        mnist_crop_root=MNIST_CROP_ROOT,
        mnist_mask_root=MNIST_MASK_ROOT,
        test_folder="testing",
        train_folder="training",
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

    :param str mnist_crop_root: root folder of cropped images (.jpg or .png)
    :param str mnist_mask_root: root folder of masks (.jpg or .png)
    :param str test_folder: folder name of
   """
    print("Loading data ...")

    # possible preparation
    if not os.path.isdir(mnist_crop_root) or not os.path.isdir(mnist_mask_root):
        crop_and_mask_mnist()

    # iterate over labels
    further_args = {}
    if not do_resize:
        further_args = {"resizefunc": lambda x: x}
    test = load_labeled_data_from_folder(test_folder, mnist_crop_root, **further_args)
    train = load_labeled_data_from_folder(train_folder, mnist_crop_root, **further_args)
    return train, test


# ------------
# RANDOM UTILS
# ------------
def random_letter(labelroot=os.path.join(MNIST_CROP_ROOT, "testing")):
    """Randomly pick an image from a random label folder.

    Needs a folder structure of:
    labelroot
      label folders
        images

    Parameters:
        str labelroot: path to the directory, where the label folders lie in

    Return:
        label, imgroot, randimgfile: label, root, and filename of the randomly chosen image
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


def random_color(max_brightness=MAX_BRIGHTNESS):
    """Picks a color out of the VALID_COLORS for the specified max_brightness.

    The set of valid colors for very encountered max_brightness value is
    saved in VALID_COLORS to be reused.

    Parameters:
        int max_brightness: maximum value of the sum of all channels;
            has to be in [0, 3*255]
    """
    if max_brightness not in VALID_COLORS.keys():
        VALID_COLORS[max_brightness] = valid_colors(max_brightness)
    randcolor = random.choice(VALID_COLORS[max_brightness])
    return randcolor


def random_window(image, window_xdim, window_ydim):
    img_ydim, img_xdim = image.shape[0], image.shape[1]
    window_aspect_ratio = window_ydim / window_xdim

    # max window dimensions with above aspect ratio
    window_maxw = min(img_xdim, int(img_ydim / window_aspect_ratio))
    # random window width, height within above bounds
    w = random.randint(window_xdim, window_maxw) if window_maxw > window_xdim else window_maxw
    h = int(window_aspect_ratio * w)
    # random position of window in texture image
    x, y = random_anchor(w, h, img_xdim, img_ydim)

    # extract window as image and resize to wanted dimensions
    window = image[y:y + h, x:x + w]
    window = cv2.resize(window, (window_xdim, window_ydim))
    return window


def random_coord(xmin, xmax, ymin, ymax):
    return (random.randint(xmin, xmax), random.randint(ymin, ymax))


def random_anchor(w, h, xdimension, ydimension,
                  occupied_boxes=None,
                  valid_box_anchors=None):
    """
    :param int w: width of the window in px
    :param int h: height of the window in px
    :param int xdimension: width of the complete image
    :param int ydimension: height of the complete image
    :param list occupied_boxes: list/tuple of boxes of the form ((x,y), (width, height))
        the window may not intersect with
    :return: (x,y) valid random coordinate
    """
    if occupied_boxes is None:
        return random.randint(0, xdimension - w), random.randint(0, ydimension - h)
    valid_box_anchors = valid_box_anchors or \
                        [(x, y)
                         for x in range(0, xdimension-w)
                         for y in range(0, ydimension-h)
                         if not do_boxes_intersect(((x, y), (w, h)), *occupied_boxes)]
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
def to_match_dict(label, bounding_box, mask):
    return {
        JSON_LABEL_KEY: label,
        JSON_BOUNDING_BOX_KEY: bounding_box,
        JSON_MASK_KEY: mask.tolist()
    }


def match_to_tuple(match, mask_resolution=JSON_MASK_RESOLUTION):
    """Reformat dict match to tuple.
    :param dict match: match with keys JSON_LABEL_KEY, JSON_BOUNDING_BOX_KEY, JSON_MASK_KEY
        and values as specified for these keys
    :param mask_resolution: resolution the mask is resized to
    :return: tuple (label, bounding box, mask in mask_resolution)
    """
    mask = np.array(match[JSON_MASK_KEY])
    # TODO: make resize work
    # if (mask.shape[1], mask.shape[0]) != mask_resolution:
    #     mask = cv2.resize(mask, mask_resolution)

    # ensure integer coordinates
    # see https://stackoverflow.com/questions/13225525/system-error-new-style-getargs-format-but-argument-is-not-a-tuple-when-using/43656642#43656642
    (x1, y1), (x2, y2) = match[JSON_BOUNDING_BOX_KEY]
    bounding_box = ((int(x1), int(y1)), (int(x2), int(y2)))

    return (
        match[JSON_LABEL_KEY],
        bounding_box,
        mask
    )


# -----------------------------
# Image generation
# -----------------------------
def generate_random_image(
        min_letters=0, max_letters=8,
        min_lettersize=28, max_lettersize=60,
        max_overlap=5,
        max_height_variance=0.2,
        xdimension=400, ydimension=300,
        max_brightness=MAX_BRIGHTNESS,
        max_alpha=0.2,
        debug=False
):
    """
    :param int min_letters, max_letters: minimal and maximum number of letters per image
    :param float max_height_variance:  in [0,1]; max percentage one letter may differ from the average size
    :param int max_overlap: maximum number of pixels letter boxes may overlap
    :param int max_brightness:  in [0, 255]; maximum brightness of one color channel
    :param int max_brightness: in [0, 3*255]; maximum brightness of all letter channels together; see random_color()
    :param max_alpha: maximum
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

    # TEXTURE
    texturefile = random_file(TEXTURES_FOLDER)
    # TODO: Why Error "Premature end of JPEG file"?t
    texture = load_image(os.path.join(TEXTURES_FOLDER, texturefile))
    # (randomly) extract window for image
    image = random_window(texture, xdimension, ydimension)

    # GRID
    image = apply_layer_from_random_file(image, layerroot=GRIDS_FOLDER)

    # (STAINS)
    image = apply_layer_from_random_file(image, layerroot=STAINS_FOLDER)

    # LETTERS
    num_letters = random.randint(min_letters, max_letters)
    if num_letters == 0:
        return image, ()

    letterfiles = list(map(lambda r: random_letter(),
                           range(0, num_letters)))
    # common parameters
    avg_height = random.randint(min_lettersize, max_lettersize)
    alpha = random.random() * max_alpha + (1 - max_alpha)  # weight for mixing in the letter
    if debug:
        print("Average height:\t", avg_height)
        print("Alpha factor:\t", alpha)

    # loop over chosen letters
    occupied_boxes = []
    for label, root, l in letterfiles:
        letter, scale = resized_image_from_file(root, l,
                                                avg_height=avg_height, max_height_variance=max_height_variance)
        mask, scale = resized_image_from_file(otherroot(root, MNIST_MASK_ROOT, image_mnist_root=MNIST_CROP_ROOT), l,
                                              fixed_scaling_factor=scale)

        # COLOR: apply random color factors to letter channels
        channel_factors = tuple(map(lambda col: (255 - col) / 255,
                                    random_color(max_brightness)))
        for i in range(0, letter.shape[2]):
            letter[:, :, i] = np.multiply(letter[:, :, 0], channel_factors[i])

        # coordinates
        w, h = letter.shape[1], letter.shape[0]
        x, y = random_anchor(w, h, xdimension, ydimension, occupied_boxes=occupied_boxes)

        # apply letter to image
        image[y:y + h, x:x + w] = cv2.addWeighted(image[y:y + h, x:x + w], 1, letter, -alpha, 0)

        # add letter mask, label, and coord. to matches
        occupied_boxes.append(((x + max_overlap, y + max_overlap), (w-2*max_overlap, h-2*max_overlap)))
        matches.append(to_match_dict(label=label,
                                     bounding_box=((x, y), (x + w, y + h)),
                                     mask=mask))
    # (STAINS)

    if debug:
        # draw metadata into image
        image = draw_masks_and_labels(image,
                                      map(lambda m: match_to_tuple(m), matches))

    return image, matches


def generate_labeled_data_files(batch_size=1000,
                                imageroot=DATA_IMAGEROOT,
                                annotationsroot="data/mask_rcnn/annotations",
                                annotations_filename_format="annotations{}.json",
                                image_filename_format = "{}.jpg",
                                start_id_enumeration=1,
                                mask_resolution=JSON_MASK_RESOLUTION,
                                num_batches=500
                                ):
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

    :param int batch_size: number of images that are created and annotated in one file per batch
    :param int num_batches: number of batches to perform;
        each batch produces one annotation file under annotationsroot
    :param str imageroot: path to folder to store images in
    :param str annotationsroot: path to folder to store annotation files in
    :param str annotations_filename_format: format string accepting the batch_id for naming the annotation files
    :param int start_id_enumeration: the filenames serve as id, and are
        enumerated sequentially starting at start_id_enumeration
    :param tuple mask_resolution: resolution as (width, height) to which the mask is resized before saving
    :return:
    """
    # ensure folders exist
    if not os.path.isdir(imageroot):
        os.makedirs(imageroot)
    if not os.path.isdir(annotationsroot):
        os.makedirs(annotationsroot)

    for batch_id in range(0, num_batches):
        annotations = []
        for img_id in range(start_id_enumeration, start_id_enumeration + batch_size):
            image, matches = generate_random_image()
            imagefilename = image_filename_format.format(img_id)
            imagefile = os.path.join(imageroot, imagefilename)

            # save image
            print(imagefile)
            write_image(imagefile, image)
            # note annotation
            annotations.append({
                JSON_FILENAME_KEY: imagefilename,
                JSON_MATCHES_KEY: matches
            })

        # save all annotations
        annotations_filename = annotations_filename_format.format(batch_id)
        annotationsfile = os.path.join(annotationsroot, annotations_filename)
        with open(annotationsfile, 'w+') as annotations_filehandle:
            print("Writing ", annotationsfile, "...")
            json.dump(annotations, annotations_filehandle)

        start_id_enumeration += batch_size


# -----------
# LOAD TOOLS
# -----------

def load_labeled_data(annotationsroot="data/mask_rcnn/annotations",
                      imageroot=DATA_IMAGEROOT,
                      mask_resolution=JSON_MASK_RESOLUTION):
    """Read in images and annotations as specified in annotationsfiles found in annotationsroot.

    :param str annotationsroot: path to root folder of json files with annotations
        of the format specified in generate_labeled_data()
    :param tuple mask_resolution: resolution as (width, height) to which the mask is resized
    :return: tuple of lists of the format
        (list of images,
        list of matches each as tuple)
        where the tuples are of the form specified in match_to_tuple()
    """
    data = []
    for root, dirs, annotationsfiles in os.walk(annotationsroot):
        for annotationsfilename in annotationsfiles:
            annotationsfile = os.path.join(root, annotationsfilename)
            with open(annotationsfile, 'r') as annotations_filehandle:
                annotations = json.load(annotations_filehandle)

            for annotation in annotations:
                # image
                imagefile = os.path.join(imageroot, annotation[JSON_FILENAME_KEY])
                print("loading image", imagefile)
                image = load_image(imagefile)

                matches = list(map(
                    lambda match: match_to_tuple(match, mask_resolution),
                    annotation[JSON_MATCHES_KEY]))
                data.append((image, matches))

    return list(zip(*data))


# -------------------
# INSPECTION TOOLS
# -------------------

def draw_bounding_boxes(image, matches):
    """
    Return the image with bounding boxes.
    :param np.array image: image
    :param iterator matches: list of matches as tuples
    :return: image with bounding boxes
    """
    for match in matches:
        cv2.rectangle(image,
                      match[1][0],  # (x1, y1)
                      match[1][1],  # (x2, y2)
                      BOUNDING_BOX_COLOR)
    return image


def draw_masks(image, matches):
    image = draw_bounding_boxes(image, matches)
    for match in matches:
        (x1, y1), (x2, y2) = match[1][0], match[1][1]
        mask = simple_resize(match[2], (x2-x1, y2-y1)).astype(image.dtype)
        image[y1:y2, x1:x2] = cv2.addWeighted(image[y1:y2, x1:x2], 1,
                                              mask, -1,
                                              0)
    return image


def draw_masks_and_labels(image, matches):
    image = draw_masks(image, matches)
    for match in matches:
        label = match[0]
        font, fontScale, thickness = cv2.FONT_HERSHEY_PLAIN, 1, 1

        # background of label
        text_width, text_height = cv2.getTextSize(label, font, fontScale, thickness)[0]
        x1, y1 = match[1][0]
        bottom_left_corner = (x1, y1+text_height)
        image[y1:y1+text_height, x1:x1+text_width] = 0

        # label
        image = cv2.putText(image,
                            label,
                            bottom_left_corner,
                            fontFace=font,
                            fontScale=fontScale,
                            color=BOUNDING_BOX_COLOR,
                            thickness=1)
    return image


if __name__ == "__main__":
    # crop_and_mask_mnist()

    # test create_random_image
    #img, tags = create_random_image(debug=True)
    #write_image("blub.png", img)

    # test generate_labeled_data()
    generate_labeled_data_files(batch_size=100, num_batches=10)
    # data = load_labeled_data()
    # for i in range(0, len(data)):
    #     img = data[0][i]
    #     matches = data[1][i]
    #     img = draw_bounding_boxes(img, matches)
    #     write_image("test"+i+".png", img)

# image sources:
# <a href='https://www.freepik.com/free-vector/wrinkled-paper-texture_851248.htm'>Designed by Freepik</a>
# <a href='https://www.freepik.com/free-vector/realistic-paper-grain-texture_923291.htm'>Designed by Freepik</a>
# <a href='https://www.freepik.com/free-photo/white-crumpled-paper-texture-for-background_1189772.htm'>Designed by Freepik</a>
# <a href='https://www.freepik.com/free-photo/dirty-pattern-paint-room-block_1088379.htm'>Designed by Freepik</a>
# <a href='https://www.freepik.com/free-vector/gradient-abstract-texture-background_1359668.htm'>Designed by Freepik</a>
# <a href="https://www.freepik.com/free-photo/white-crumpled-paper-texture-for-background_1189772.htm">Designed by Freepik</a>
# <a href="https://www.freepik.com/free-vector/realistic-coffee-cup-stain-collection_1577305.htm">Designed by Freepik</a>
