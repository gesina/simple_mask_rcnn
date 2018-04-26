import keras
import keras.models as km
import keras.layers as kl
import keras.engine as ke
import keras.backend as kb
import tensorflow as tf
import numpy as np


class Config:
    # DEFAULTS

    IMAGE_MATCHES_SHAPE = ()

    # number of different anchor shapes per center
    NUM_ANCHOR_SHAPES = 3

    # number of proposals to be processed for masks;
    # should be higher than the the maximum number of objects with bounding boxes per image!
    NUM_PROPOSALS = 12

    # number of proposals to be processed before NMS,
    # selected by best foreground score
    PRE_NMS_LIMIT = 30

    # threshold of foreground score for Non-Maximum Suppression
    NMS_THRESHOLD = 0.7,

    IMAGE_SHAPE = (None, None, 3),

    RPN_CLS_LOSS_NAME = "rpn_cls_loss"
    RPN_REG_LOSS_NAME = "rpn_reg_loss"
    LOSS_LAYER_NAMES = [
        RPN_CLS_LOSS_NAME,
        RPN_REG_LOSS_NAME
    ]
    LOSS_WEIGHTS = {
        RPN_CLS_LOSS_NAME: 1,
        RPN_REG_LOSS_NAME: 1
    }
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    GRADIENT_CLIP_NORM = 5.0
    METRICS = ['accuracy']


class MaskRCNNWrapper:
    """"""

    def __init__(self, config, **kwargs):
        """
        :param Config config: configuration
        """
        self.config = config
        self.model = self.build_mask_rcnn_model(**kwargs)

    def build_mask_rcnn_model(self, **kwargs):
        """
        :param dict kwargs: further named parameters for keras.models.Model()
        :return: Mask R-CNN keras model
        """
        input_image = kl.Input(shape=self.config.IMAGE_SHAPE, name="input_image")
        input_image_matches = kl.Input(shape=self.config.IMAGE_MATCHES_SHAPE, name="input_image_matches")

        # BACKBONE
        backbone = build_backbone_model(input_image)

        # REGION PROPOSALS: objectness score and coordinate correction for each anchor
        rpn_cls, rpn_cls_logits, rpn_reg = build_rpn_model(backbone, self.config.NUM_ANCHOR_SHAPES)
        rpn_merged = kl.concatenate([rpn_cls, rpn_reg])
        rpn_proposals = ProposalLayer(
            num_proposals=self.config.NUM_PROPOSALS,
            pre_nms_limit=self.config.PRE_NMS_LIMIT,
            nms_threshold=self.config.NMS_THRESHOLD,
        )

        # GROUND TRUTH BOX INPUTS
        # shape: [batch_size, num_proposals, 1: objectness class in +1/0/-1]
        input_rpn_cls_gt = kl.Input(shape=[None, None, 1], name="input_rpn_cls_gt", dtype=tf.int32)
        # all ground-truth positive and coordinate corrected anchor boxes per image;
        # in the same order (skipping negative and neutral ones) as in rpn_proposals; padded with 0s
        # shape: [batch_size, max_num_pos_anchors, 4: coord]
        input_rpn_reg_gt = kl.Input(shape=[None, None, 4], name="input_rpn_reg", dtype=tf.float32)

        # LOSS LAYERS
        rpn_cls_loss = kl.Lambda(lambda x: rpn_cls_loss_fn(*x),
                                 name=self.config.RPN_CLS_LOSS_NAME)([input_rpn_cls_gt, rpn_cls_logits])
        rpn_reg_loss = kl.Lambda(lambda x: rpn_reg_loss_fn(*x),
                                 name=self.config.RPN_REG_LOSS_NAME)([input_rpn_cls_gt, input_rpn_reg_gt, rpn_reg])

        # MODEL
        model = km.Model(inputs=[input_image, input_rpn_cls_gt, input_rpn_reg_gt],
                         outputs=[rpn_cls_loss, rpn_reg_loss],
                         **kwargs)
        return model

    def compile_model(self):
        # OPTIMIZER
        optimizer = keras.optimizers.SGD(
            lr=self.config.LEARNING_RATE,
            momentum=self.config.LEARNING_MOMENTUM,
            weight_decay=self.config.WEIGHT_DECAY,
            clipnorm=self.config.GRADIENT_CLIP_NORM
        )

        # LOSSES
        # In case weights should be applied (mind: the losses are already mean values)
        loss_layer_outputs = [self.model.get_layer(ln).output * self.config.LOSS_WEIGHTS[ln]
                              for ln in self.config.LOSS_LAYER_NAMES]

        self.model.add_loss(losses=loss_layer_outputs)
        self.model.compile(optimizer=optimizer,
                           loss=[None] * len(self.model.outputs),
                           metrics=self.config.METRICS)


def rpn_cls_loss_fn(rpn_cls_gt, rpn_cls_logits):
    """Log loss of anchor objectness classification.
    The ground-truth anchor objectness class is one of
        * 1: object
        * -1: no object
        * 0: neutral

    :param np.array rpn_cls_gt: [batch_size, num_anchors, 1] anchor ground-truth objectness class
    :param np.array rpn_cls_logits: [batch_size, num_anchors, 2: (bg, fg)] anchor bg, fg score
    :return: np.array [1,]: loss
    """
    # Squeeze last dim
    # output_shape: [batch_size, num_anchors: anchor classes]
    rpn_cls_gt = tf.squeeze(rpn_cls_gt, -1)

    # PICK NON-NEUTRAL ANCHORS
    # Only +1/-1 ground-truth anchors contribute to the loss,
    # neutral anchors (gt value = 0) don't.
    non_neutral_indices = tf.where(kb.not_equal(rpn_cls_gt, 0))  # output_shape: [batch_size, num_anchors: bools]
    rpn_cls_gt = tf.gather_nd(rpn_cls_gt, non_neutral_indices)
    rpn_class_logits = tf.gather_nd(rpn_cls_logits, non_neutral_indices)

    # CONVERT the +1/-1 ground-truth values to 0/1 values.
    rpn_cls_gt = kb.cast(kb.equal(rpn_cls_gt, 1), tf.int32)

    # CROSSENTROPY LOSS
    loss = kb.sparse_categorical_crossentropy(target=rpn_cls_gt,
                                              output=rpn_class_logits,
                                              from_logits=True)
    # MEAN
    loss = kb.switch(tf.size(loss) > 0, kb.mean(loss), tf.constant(0.0))
    return loss


def rpn_reg_loss_fn(rpn_cls_gt, rpn_reg_gt, rpn_reg):
    """Give the smooth L1-loss of all ground-truth positive anchors' coordinates.
    The ground-truth anchor objectness class is one of
        * 1: positive (contains object)
        * -1: negative (no object)
        * 0: neutral

    :param np.array rpn_cls_gt: [batch_size, num_anchors, 1]
        anchor ground-truth objectness class
    :param np.array rpn_reg_gt: [batch_size, num_anchors, 4: (x1, y1, x2, y2)]
        ground-truth bounding box coord;
        same order as ground-truth positive predicted bounding boxes, padded with 0s
    :param np.array rpn_reg: [batch_size, num_anchors, 4: (x1, y1, x2, y2)]
        predicted bounding box coord
    :return:
    """
    # PICK POSITIVE ANCHORS from rpn_reg
    # Only +1 ground-truth anchors contribute to the loss.
    # output_shape: [batch_size, num_anchors: objectness score]
    rpn_cls_gt = kb.squeeze(rpn_cls_gt, -1)
    # output_shape: [num_pos_anchors, 2: (batch idx, anchor idx)]
    non_neutral_indices = tf.where(kb.equal(rpn_cls_gt, 1))
    # output_shape: [num_pos_anchors, 4: (x1, y1, x2, y2)]
    rpn_reg = tf.gather_nd(rpn_reg, non_neutral_indices)

    # PICK POSITIVE ANCHORS from 0-padded rpn_reg_gt
    # Remove padding from rpn_reg_gt and flatten
    # output: [batch_size, 1: num_pos_anchors for this batch]
    batch_counts = kb.sum(kb.cast(kb.equal(rpn_cls_gt, 1), tf.int32), axis=1)
    # output: [num_pos_anchors, 4: (x1, y1, x2, y2)]
    rpn_reg_gt = batch_pack(rpn_reg_gt, batch_counts)

    return smooth_l1_loss(rpn_reg, rpn_reg_gt)


def smooth_l1_loss(x_pred, x_true):
    """Gives the smooth L1 loss for lists of coordinates x and ground-truth coord x_gt.
    The smooth absolute value of a real number r is defined as
    `|r|_sl1 := (|r| > 1) ? |r|-0.5 : 0.5r**2` where |r| is the L1-norm (absolute value).
    The smooth L1-norm for a coordinate d=(r1, r2,..., rn) is defined as `sum_i(|ri|_sl1)`.
    The smooth L1-loss is defined as `mean(sum(|x_true-x_pred|_sl1))`.
    It is claimed to be less sensitive to outliers than L2.

    :param np.array x_pred: [N, coord] predicted coordinates
    :param x_true: [N, coord] ground-truth coordinates
    :return: [1,] loss
    """
    diff = kb.abs(x_true - x_pred)
    less_than_one = kb.cast(kb.less(diff, 1.0), "float32")

    # SMOOTH L1 LOSS in each coord:
    # |d|_sl1 := (|d| > 1) ? |d|-0.5 : 0.5d**2
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)

    # MEAN
    loss = kb.switch(tf.size(loss) > 0, kb.mean(loss), tf.constant(0.0))
    return loss


def batch_pack(inputs, counts):
    """Flatten tensor along axis=0 (rows),
    for each row picking only the first counts[row] columns.

    :param nd.array inputs: [num_rows,] + inputs.shape[1:] tensor to be flattened and cropped
    :param counts: [num_rows, 1: num cols for this row]
    :return: [sum(counts),] + inputs.shape[1:]
    """
    outputs = []
    num_rows = inputs.shape[0]
    for i in range(0, num_rows):
        # add list of first counts[i] columns of ith row
        outputs.append(inputs[i, :counts[i]])
    # concat all lists to one list of columns
    return tf.concat(outputs, axis=0)


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
        proposals = []
        # TODO: use a function per_datum_in_batch instead of looping
        # Iterate over every datum in batch
        batch_size = tf.shape(input[0])[1]
        for datum_idx in range(0, batch_size):
            # GATHER ONE DATUM
            # output_shape: ([num_anchors, scores], [num_anchors, coord corr], [num_anchors, coord])
            datum = (inputs[0][datum_idx], inputs[1][datum_idx], inputs[2][datum_idx])

            # TRIMMING
            # Improve performance by trimming to max. pre_nms_limit number of best anchors
            # (sorted by fg score) and doing the rest on the smaller subset of size N.
            # output_shape: ([N, 2: scores], [N, 4: coord corr], [N, 4: coord])
            datum = self.trim_to_top_anchors(datum)

            # VARIABLES
            rpn_cls = datum[0]  # Box Scores
            fg_scores = rpn_cls[:, 1]  # Foreground class confidence of shape [N,]
            rpn_reg = datum[1]  # Box coordinate correcition deltas of shape [N, 4]
            anchors = datum[2]  # Anchor box coordinates of shape [N, 4]

            # REFINE ANCHORS
            # Apply coordinate_deltas to anchors to get refined anchor boxes.
            # output_shape: [N, 4: (x1, y1, x2, y2)]
            boxes = self.apply_box_deltas(anchors, rpn_reg)

            # CLIP TO IMAGE BOUNDARIES
            # Since we're in normalized coordinates, clip to 0..1 range.
            window = np.array((0, 0, 1, 1), dtype=np.float32)
            # output_shape: [N, 4: (x1, y1, x2, y2)]
            boxes = self.clip_boxes(boxes, window)

            # NON-MAXIMUM SUPPRESSION
            # Sort by fg_score, and discard boxes that overlap with higher scored
            # boxes too much to get n<=N, max. num_proposals, boxes.
            indices_to_keep = tf.image.non_max_suppression(
                boxes,
                fg_scores,
                max_output_size=self.num_proposals,
                iou_threshold=self.nms_threshold,
                name="rpn_non_max_suppression")
            # output_shape: [n, 4: (x1, y1, x2, y2)]
            datum_proposals = tf.gather(boxes, indices_to_keep)

            # PAD WITH 0s if not enough proposals are available
            # output_shape: [datum_idx+1: batch, self.num_proposals, 4: coord]
            datum_proposals = self.pad_proposals(datum_proposals)

            proposals.append(datum_proposals)

        return tf.stack(proposals, name="stack_proposals")

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

        All coordinates normalized.
        :param np.array boxes: [N, 4: (x1, y1, x2, y2)] boxes to update
        :param np.array deltas: [N, 4: (dx, dy, log(dw), log(dh))] refinements to apply
        :return: np.array [N, 4: (x1, y1, x2, y2)] updated box coordinates
        """
        # Convert to lists with x, y, w, h each of shape [N,]
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        center_xs = boxes[:, 0] + 0.5 * widths
        center_ys = boxes[:, 1] + 0.5 * heights

        # Apply deltas
        center_xs += deltas[:, 0] * widths
        center_ys += deltas[:, 1] * heights
        widths *= tf.exp(deltas[:, 2])
        heights *= tf.exp(deltas[:, 3])

        # Convert back to x1s, y1s, x2s, y2s
        x1s = center_xs - 0.5 * widths
        y1s = center_ys - 0.5 * heights
        x2s = x1s + widths
        y2s = y1s + heights

        # Concat back to box coordinates tensor
        # output_shape: [N, 4]
        result_coordinates = tf.stack([x1s, y1s, x2s, y2s],
                                      axis=1, name="apply_box_deltas_out")
        return result_coordinates

    @staticmethod
    def clip_boxes(boxes, window):
        """Trim boxes to fit into window.
        :param np.array boxes: [N, 4: (x1, y1, x2, y2)]
        :param np.array window: [4: (x1, y1, x2, y2)]
        :return: np.array [N, 4: (x1, y1, x2, y2)] updated box coordinates
        """
        # Split into lists of coordinate values
        # output_shape: 4 x [N,]
        x1s, y1s, x2s, y2s = tf.split(boxes, 4, axis=1)
        win_x1s, win_y1s, win_x2s, win_y2s = tf.split(window, 4)

        # Clip
        x1s = tf.maximum(tf.minimum(x1s, win_x2s), win_x1s)  # overlap right: right border, overlap left: left border
        y1s = tf.maximum(tf.minimum(y1s, win_y2s), win_y1s)
        x2s = tf.maximum(tf.minimum(x2s, win_x2s), win_x1s)
        y2s = tf.maximum(tf.minimum(y2s, win_y2s), win_y1s)
        # output_shape: [N, 4]
        clipped_coordinates = tf.stack([y1s, x1s, y2s, x2s],
                                       axis=1, name="clipped_boxes")
        return clipped_coordinates

    def pad_proposals(self, proposals):
        """Pad proposals with 0s until they have length self.num_proposals.

        :param np.array proposals: [n, 4] list of proposed box coordinates
        :return: np.array [self.num_proposals, 4], proposals padded with 0s at the end
        """
        padding = tf.maximum(self.num_proposals - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
        return proposals
