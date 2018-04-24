import keras.models as km
import keras.layers as kl
import keras.engine as ke
import tensorflow as tf
import numpy as np


def build_mask_rcnn_model(
        image_matches_shape,
        num_anchor_shapes,
        num_proposals,
        pre_nms_limit,
        nms_threshold=0.7,
        image_shape=(None, None, 3)
):
    """

    :param image_matches_shape:
    :param num_anchor_shapes: number of different anchor shapes per center
    :param int num_proposals: number of proposals to be processed for masks
    :param int pre_nms_limit: number of proposals to be processed before NMS,
        selected by best foreground score
    :param float nms_threshold: threshold of foreground score for Non-Maximum Suppression
    :param image_shape:
    :return:
    """
    input_image = kl.Input(shape=image_shape, name="input_image")
    input_image_matches = kl.Input(shape=image_matches_shape, name="input_image_matches")

    # BACKBONE
    backbone = build_backbone_model(input_image)

    # REGION PROPOSALS: objectness score and coordinate correction for each anchor
    rpn_cls, rpn_cls_logits, rpn_reg = build_rpn_model(backbone, num_anchor_shapes)
    rpn_merged = kl.concatenate([rpn_cls, rpn_reg])
    rpn_rois = ProposalLayer(
        num_proposals=num_proposals,
        pre_nms_limit=pre_nms_limit,
        nms_threshold=nms_threshold,
    )

    # GROUND TRUTH BOX INPUTS
    input_rpn_match = kl.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
    # TODO: What is this meant to be?
    input_rpn_reg = kl.Input(shape=[None, 4], name="input_rpn_reg", dtype=tf.float32)

    # LOSSES
    loss_rpn_cls = kl.Lambda(lambda x: rpn_cls_loss(*x),
                             name="loss_rpn_cls")([input_rpn_match, rpn_cls_logits])
    loss_rpn_reg = kl.Lambda(lambda x: rpn_reg_loss(*x),
                             name="loss_rpn_reg")([input_rpn_reg, input_rpn_match, rpn_reg])


def rpn_cls_loss():


def rpn_reg_loss():


def build_rpn_model(model_input, anchors_per_location):
    """Build the RPN model.
    It consists of
        - a shared model
        - a model for objectness prediction (cls) per anchor shape, and
        - a model for box prediction (reg) per anchor shape.
    The return consists of
        - *rpn_cls* (class predictions, shape [batch_size, anchors, 2]):
            giving a foreground and a background probability for each anchor.
        - *rpn_reg* (bounding box refinements, shape [batch_size, anchors, 4]):
            giving a coordinate correction (dx, dy, log(dw), log(dh)) per anchor.

    :param model_input: backbone model output
    :param int anchors_per_location: anchors per region center
    :return: rpn_cls, rpn_reg
    """
    # SHARED MODEL for cls and reg
    shared = build_rpn_shared_model(model_input)

    # ANCHOR SCORES: cls model
    # Each anchor gets two filters, one for fg score, one for bg score
    # output_shape: [batch_size, anchors, 2]
    rpn_cls, rpn_cls_logits = build_rpn_cls_model(anchors_per_location, shared)

    # BOUNDING BOX REFINEMENT: reg model
    # Each anchor gets 4 filters, each for one part of the coordinate correction:
    # dx, dy, log(dw), log(dh)
    # output_shape: [batch_size, anchors, 4]
    rpn_reg = build_rpn_reg_model(anchors_per_location, shared)

    return rpn_cls, rpn_cls_logits, rpn_reg


def build_rpn_reg_model(anchors_per_location, shared):
    # Each anchor gets 4 filters, each for one part of the coordinate correction:
    # dx, dy, log(dw), log(dh)
    # output_shape: [batch_size, height, width, anchors_per_location, depth=[dx,dy,log(dw),log(dh)]]
    bbox_refinement_output = kl.Conv2D(
        filters=anchors_per_location * 4,
        kernel_size=(1, 1),
        padding="valid",
        activation='linear',
        name='rpn_bbox_pred'
    )(shared)

    # Resize
    # output_shape: [batch_size, anchor_filterquatuples, 4 (x,y,log(w),log(h))]
    rpn_reg = kl.Lambda(
        lambda t: tf.reshape(
            t, (
                tf.shape(t)[0],
                -1,
                4)
        )
    )(bbox_refinement_output)

    return rpn_reg


def build_rpn_cls_model(anchors_per_location, model_input):
    """Model for getting anchor scores from rpn_shared_model."""
    # Each anchor gets two filters, one for fg score, one for bg score
    # output_shape: [batch_size, height, width, anchors_per_location*2]
    anchor_scores_output = kl.Conv2D(
        filters=2 * anchors_per_location,
        kernel_size=(1, 1),
        padding='valid',
        activation='linear',
        name='rpn_class_raw'
    )(model_input)

    # Resize
    # output_shape: [batch_size, anchor_filterpairs, 2]
    rpn_cls_logits = kl.Lambda(
        lambda t: tf.reshape(
            t, (tf.shape(t)[0],  # batch_size
                -1,  # rest
                2)  # fg/bg score
        )
    )(anchor_scores_output)

    # Softmax: produce cls prediction
    # output_shape: as above
    rpn_cls = kl.Activation(
        "softmax", name="rpn_cls")(rpn_cls_logits)

    return rpn_cls, rpn_cls_logits


def build_rpn_shared_model(model_input):
    shared_output = kl.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",  # same takes ridiculously much longer than valid...
        activation="relu",
        name='rpn_conv_shared'
    )(model_input)
    return shared_output


def build_backbone_model(image_input):
    """Build convolutional backbone model.

    :param tuple image_input: keras.layers.Input(); shape should be divisible by 2**3
    :return: backbone_model as layer output
    """
    backbone_output = kl.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",  # same takes ridiculously much longer than valid...
        activation="relu",
    )(image_input)
    backbone_output = kl.MaxPooling2D(pool_size=(2, 2))(backbone_output)

    backbone_output = kl.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",  # same takes ridiculously much longer than valid...
        activation="relu",
    )(backbone_output)
    backbone_output = kl.MaxPooling2D(pool_size=(2, 2))(backbone_output)

    backbone_output = kl.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        padding="valid",
        activation="relu",
    )(backbone_output)
    backbone_output = kl.MaxPooling2D(pool_size=(2, 2))(backbone_output)

    backbone_output = kl.Dropout(rate=0.25)(backbone_output)

    # backbone_model = km.Model(input=image_input, output=backbone_output)
    return backbone_output


class ProposalLayer(ke.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor foreground
    scores and non-max suppression to remove overlaps. It also applies
    bounding box refinement deltas to anchors.
    The output will be self.num_proposals, possibly padded with 0s in
    case not enough good proposals are available.
    Inputs: See call()
    """

    def __init__(self,
                 num_proposals,
                 pre_nms_limit,
                 nms_threshold,
                 rpn_bbox_standard_dev=np.array((0.1, 0.1, 0.2, 0.2)),
                 bbox_standard_dev=np.array((0.1, 0.1, 0.2, 0.2)),
                 config=None, **kwargs):
        """
        :param int num_proposals: number of proposals to return;
            also maximum number of proposals allowed for NMS;
            padded with 0s if not enough proposals are available
        :param int pre_nms_limit: maximum number of best choice boxes considered for NMS;
            should be larger than num_proposals!
        :param nms_threshold: IoU threshold for NMS
        :param kwargs: e.g. name
        :param np.array rpn_bbox_standard_dev: bounding box refinement standard deviation for RPN
        :param np.array bbox_standard_dev: bounding box refinement standard deviation for final detections
        :param config: further layer config
        """
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.num_proposals = num_proposals
        self.nms_threshold = nms_threshold
        self.pre_nms_limit = pre_nms_limit
        self.rpn_bbox_standard_dev = rpn_bbox_standard_dev
        self.bbox_standard_dev = bbox_standard_dev

    def compute_output_shape(self, input_shape):
        return [None, self.num_proposals, 4]

    def call(self, inputs, **kwargs):
        """Choose proposal boxes using foreground score with non-maximum suppression.

        :param inputs: consists of (all coordinates normalised):
            - *rpn_cls*: [batch, anchors, (bg prob, fg prob)]
            - *rpn_reg*: [batch, anchors, (dx, dy, log(dw), log(dh))]
            - *anchors*: [batch, (x1, y1, x2, y2)] anchors in normalized coordinates
        :return: List of proposals in normalized coordinates,
            of shape [batch, rois, (x1, y1, x2, y2)], and length self.num_proposals
            (possibly padded with 0s)
        """
        # TRIMMING
        # Improve performance by trimming to pre_nms_limit number of best anchors
        # (sorted by fg score) and doing the rest on the smaller subset.
        self.trim_inputs(inputs)

        # VARIABLES
        rpn_cls = inputs[0]  # Box Scores
        fg_scores = rpn_cls[:, :, 1]  # foreground class confidence. [Batch, num_rois, 1]
        # TODO: Understand bounding box deltas
        # Box coordinate_deltas [batch, num_rois, 4]
        # (dx, dy, log(dw), log(dh)) -> (dx*0.1, dy*0.1, log(dw)*0.2, log(dh)*0.2)
        rpn_reg = inputs[1] * np.reshape(self.rpn_bbox_standard_dev, [1, 1, 4])
        anchors = inputs[2]  # Anchors

        # REFINED ANCHORS
        # Apply coordinate_deltas to anchors to get refined anchor boxes.
        # output_shape: [batch, N, (y1, x1, y2, x2)]
        boxes = self.apply_box_deltas(anchors, rpn_reg)

        # CLIP TO IMAGE BOUNDARIES
        # Since we're in normalized coordinates, clip to 0..1 range.
        # output_shape: [batch, N, (y1, x1, y2, x2)]
        window = np.array((0, 0, 1, 1), dtype=np.float32)
        boxes = self.clip_boxes(boxes, window)

        # NON-MAXIMUM SUPPRESSION
        indices_to_keep = tf.image.non_max_suppression(
            boxes,
            fg_scores,
            max_output_size=self.num_proposals,
            iou_threshold=self.nms_threshold,
            name="rpn_non_max_suppression")
        proposals = tf.gather(boxes, indices_to_keep)

        # PAD WITH 0s if not enough proposals are available
        proposals = self.pad_proposals(proposals)

        return proposals

    def trim_inputs(self, inputs):
        """Trim inputs to the top self.pre_nms_limit number of anchors,
        measured by foreground score."""
        batch_size = tf.shape(input[0])[1]
        pre_nms_limit = tf.minimum(self.pre_nms_limit, batch_size)
        # Iterate over every datum in batch
        for datum_idx in range(0, batch_size):
            batch_fg_scores = inputs[0][datum_idx, :, 1]
            # top k anchor indices
            indices_to_keep = tf.nn.top_k(batch_fg_scores,
                                          k=pre_nms_limit,
                                          sorted=True, name="top_anchors").indices
            # trim data to the above top k anchors
            for i in range(0, len(inputs)):
                inputs[i][datum_idx, :, :] = \
                    tf.gather(inputs[i][datum_idx, :, :],
                              indices=indices_to_keep,
                              axis=1)  # anchor axis

    @staticmethod
    def apply_box_deltas(boxes, deltas):
        """Applies the given deltas to the given boxes.
        :param boxes: [N, (x1, y1, x2, y2)] boxes to update
        :param deltas: [N, (dx, dy, log(dw), log(dh))] refinements to apply
        """
        # Convert to x, y, w, h
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        center_x = boxes[:, 0] + 0.5 * width
        center_y = boxes[:, 1] + 0.5 * height

        # Apply deltas
        center_x += deltas[:, 0] * width
        center_y += deltas[:, 1] * height
        width *= tf.exp(deltas[:, 2])
        height *= tf.exp(deltas[:, 3])

        # Convert back to x1, y1, x2, y2
        x1 = center_x - 0.5 * width
        y1 = center_y - 0.5 * height
        x2 = x1 + width
        y2 = y1 + height

        # Name
        result = tf.stack([x1, y1, x2, y2], axis=1, name="apply_box_deltas_out")
        return result

    @staticmethod
    def clip_boxes(boxes, window):
        """Trim boxes to fit into window.
        :param boxes: [N, (x1, y1, x2, y2)]
        :param window: [4] in the form (x1, y1, x2, y2)
        """
        # Split
        x1, y1, x2, y2 = tf.split(boxes, 4, axis=1)  # lists of coordinate values
        wx1, wy1, wx2, wy2 = tf.split(window, 4)

        # Clip
        x1 = tf.maximum(tf.minimum(x1, wx2), wx1)  # overlap right: right border, overlap left: left border
        y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
        x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
        y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
        clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
        clipped.set_shape((clipped.shape[0], 4))
        return clipped

    def pad_proposals(self, proposals):
        """Pad proposals with 0s until they have length self.num_proposals."""
        padding = tf.maximum(self.num_proposals - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
        return proposals
