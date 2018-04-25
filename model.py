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
    return 1


def rpn_reg_loss():
    return 1


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

        :param tuple inputs: consists of np.arrays (all coordinates normalised):
            - *rpn_cls*: [batch_size, num_anchors, 2 scores (bg prob, fg prob)]
            - *rpn_reg*: [batch_size, num_anchors, 4 coord. corrections (dx, dy, log(dw), log(dh))]
            - *anchors*: [batch_size, 4 coord. (x1, y1, x2, y2)]
        :return: List of proposals in normalized coordinates,
            of shape [batch_size, num_proposals, (x1, y1, x2, y2)]
            (possibly padded with 0s)
        """
        # TRIMMING
        # Improve performance by trimming to pre_nms_limit number of best anchors
        # (sorted by fg score) and doing the rest on the smaller subset.

        # TODO: Datatype ndarray?
        proposals = []
        # Iterate over every datum in batch
        batch_size = tf.shape(input[0])[1]
        for datum_idx in range(0, batch_size):
            # datum = ((anchors, scores), (anchors, coord corr), (anchors, coord))
            datum = (inputs[0][datum_idx], inputs[1][datum_idx], inputs[2][datum_idx])
            datum = self.trim_to_top_anchors(datum)

            # VARIABLES
            rpn_cls = datum[0]  # Box Scores
            fg_scores = rpn_cls[:, 1]  # foreground class confidence. [Batch, num_rois, 1]
            # TODO: Understand bounding box deltas
            # Box coordinate_deltas of shape [num_rois, 4]; do
            # (dx, dy, log(dw), log(dh)) -> (dx*0.1, dy*0.1, log(dw)*0.2, log(dh)*0.2)
            rpn_reg = datum[1] * np.reshape(self.rpn_bbox_standard_dev, [1, 4])
            anchors = datum[2]

            # REFINED ANCHORS
            # Apply coordinate_deltas to anchors to get refined anchor boxes.
            # output_shape: [num_anchors, (y1, x1, y2, x2)]
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
            datum_proposals = tf.gather(boxes, indices_to_keep)

            # PAD WITH 0s if not enough proposals are available
            datum_proposals = self.pad_proposals(datum_proposals)

            # TODO: tf.concat?
            proposals.append(datum_proposals)

        return proposals

    def trim_to_top_anchors(self, datum):
        """Trim anchors for datum to the top self.pre_nms_limit number of anchors,
        measured by foreground score.

        :param tuple datum:
            (
                np.array of shape [anchors, 2 scores (bg, fg)],
                np.array of shape [anchors, 4 coord corr (dx, dy, log(dw), log(dh)],
                np.array of shape [anchors, 4 coord (x1, y1, x2, y2)]
            )"""

        anchor_axis = 0
        other_data_axis = 1
        batch_fg_scores = datum[0][:, other_data_axis]
        num_anchors = tf.shape(datum[2])[anchor_axis]
        pre_nms_limit = tf.minimum(self.pre_nms_limit, num_anchors)

        # top k anchor indices
        indices_to_keep = tf.nn.top_k(batch_fg_scores,
                                      k=pre_nms_limit,
                                      sorted=True, name="top_anchors").indices

        # trim all parts of the datum to the above top k anchors
        return tuple(map(
            lambda tensor: tf.gather(tensor,
                                     indices=indices_to_keep,
                                     axis=anchor_axis),
            datum
        ))

    @staticmethod
    def apply_box_deltas(boxes, deltas):
        """Applies the given deltas to the given boxes.
        :param np.array boxes: [N, 4: (x1, y1, x2, y2)] boxes to update
        :param np.array deltas: [N, 4: (dx, dy, log(dw), log(dh))] refinements to apply
        """
        # Convert to lists with x, y, w, h each of length N
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        center_xs = boxes[:, 0] + 0.5 * widths
        center_ys = boxes[:, 1] + 0.5 * heights

        # Apply deltas
        center_xs += deltas[:, 0] * widths
        center_ys += deltas[:, 1] * heights
        widths *= tf.exp(deltas[:, 2])
        heights *= tf.exp(deltas[:, 3])

        # Convert back to x1, y1, x2, y2
        x1s = center_xs - 0.5 * widths
        y1s = center_ys - 0.5 * heights
        x2s = x1s + widths
        y2s = y1s + heights

        # Name
        result_coordinates = tf.stack([x1s, y1s, x2s, y2s],
                                      axis=1, name="apply_box_deltas_out")
        return result_coordinates

    @staticmethod
    def clip_boxes(boxes, window):
        """Trim boxes to fit into window.
        :param np.array boxes: [N, 4: (x1, y1, x2, y2)]
        :param np.array window: [4: (x1, y1, x2, y2)]
        """
        # Split into lists of coordinate values
        x1s, y1s, x2s, y2s = tf.split(boxes, 4, axis=1)
        win_x1s, win_y1s, win_x2s, win_y2s = tf.split(window, 4)

        # Clip
        x1s = tf.maximum(tf.minimum(x1s, win_x2s), win_x1s)  # overlap right: right border, overlap left: left border
        y1s = tf.maximum(tf.minimum(y1s, win_y2s), win_y1s)
        x2s = tf.maximum(tf.minimum(x2s, win_x2s), win_x1s)
        y2s = tf.maximum(tf.minimum(y2s, win_y2s), win_y1s)
        clipped_coordinates = tf.stack([y1s, x1s, y2s, x2s],
                                       axis=1, name="clipped_boxes")
        return clipped_coordinates

    def pad_proposals(self, proposals):
        """Pad proposals with 0s until they have length self.num_proposals."""
        padding = tf.maximum(self.num_proposals - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
        return proposals
