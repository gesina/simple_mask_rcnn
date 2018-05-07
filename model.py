import keras
import keras.models as km
import keras.layers as kl
import keras.engine as ke
import keras.backend as kb
import keras.callbacks as kc
import tensorflow as tf
import numpy as np


def smooth_l1_loss(x_pred, x_true, normalization_factor=None):
    """Gives the normalized, smooth L1 loss for lists of coordinates x and ground-truth coord x_gt.
    The smooth absolute value of a real number r is defined as
    `|r|_sl1 := (|r| > 1) ? |r|-0.5 : 0.5r**2` where |r| is the L1-norm (absolute value).
    The smooth L1-norm for a coordinate d=(r1, r2,..., rn) is defined as `sum_i(|ri|_sl1)`.
    The smooth L1-loss is defined as `mean(sum(|x_true-x_pred|_sl1))`.
    It is claimed to be less sensitive to outliers than L2.

    :param np.array x_pred: [N, coord] predicted coordinates
    :param x_true: [N, coord] ground-truth coordinates
    :param float normalization_factor: factor to apply to sum of losses
    instead of taking the mean
    :return: [1,] loss
    """
    diff = kb.abs(x_true - x_pred)
    less_than_one = kb.cast(kb.less(diff, 1.0), 'float32')

    # SMOOTH L1 LOSS in each coord:
    # |d|_sl1 := (|d| > 1) ? |d|-0.5 : 0.5d**2
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)

    # SUM and NORMALIZE BY normalizer
    if normalization_factor is None:
        loss = kb.switch(tf.size(loss) > 0, kb.mean(loss), tf.constant(0.0))
    else:
        loss = tf.reduce_sum(loss) * normalization_factor
    return loss


class MaskRCNNWrapper:
    """"""

    def __init__(self, config, train=True, **kwargs):
        """
        :param Config config: configuration
        """
        self.loss_fn = smooth_l1_loss
        self.config = config
        self.training_mode = train
        if train:
            print("Building pretraining model ...")
            self.backbone_pretraining_model = self.build_backbone_pretraining_model()
        print("Building training model ...")
        self.model = self.build_mask_rcnn_model(train, **kwargs)

    def build_mask_rcnn_model(self, train=True, **kwargs):
        """
        :param boolean train: if yes, return training model (including loss layers)
        :param dict kwargs: further named parameters for keras.models.Model()
        :return: Mask R-CNN keras model, either for training (with losses) or for inference
        """
        # INPUTS
        input_image = kl.Input(shape=self.config.IMAGE_SHAPE, name="input_image")
        # input_image_matches = kl.Input(shape=self.config.IMAGE_MATCHES_SHAPE, name="input_image_matches")

        # BACKBONE
        # Ground-truth box inputs
        # shape: [batch_size, num_proposals, 1: objectness class in +1/0/-1]
        input_rpn_cls_gt = kl.Input(shape=self.config.RPN_CLS_SHAPE,
                                    name="input_rpn_cls_gt", dtype=tf.int32)
        # All ground-truth positive and coordinate corrected anchor boxes per image;
        # in the same order (skipping negative and neutral ones) as in rpn_proposals; padded with 0s
        # shape: [batch_size, max_num_pos_anchors, 4: coord]
        input_rpn_reg_deltas_gt = kl.Input(shape=self.config.RPN_REG_SHAPE,
                                           name="input_rpn_reg", dtype=tf.float32)
        input_anchors = kl.Input(shape=[None, 4], name="input_anchors")

        # COMMON BACKBONE
        backbone = self.build_backbone_model(input_image)

        # REGION PROPOSALS: objectness score and coordinate correction for each anchor
        rpn_cls, rpn_cls_logits, rpn_reg_deltas = \
            self.build_rpn_model(model_input=backbone,
                                 anchors_per_location=self.config.NUM_ANCHOR_SHAPES)

        # LOSS LAYERS
        rpn_cls_loss = kl.Lambda(lambda x: self.rpn_cls_loss_fn(*x),
                                 name=self.config.RPN_CLS_LOSS_NAME)([input_rpn_cls_gt, rpn_cls_logits])
        rpn_reg_loss = kl.Lambda(lambda x: self.rpn_reg_loss_fn(*x),
                                 name=self.config.RPN_REG_LOSS_NAME)(
            [input_rpn_cls_gt, input_rpn_reg_deltas_gt, rpn_reg_deltas])

        # REGION PROPOSALS: select the best ones and turn deltas to coordinates
        rpn_merged = kl.concatenate([rpn_cls, rpn_reg_deltas, input_anchors],
                                    axis=2)
        rpn_proposals = ProposalLayer(
            num_proposals=self.config.NUM_PROPOSALS,
            pre_nms_limit=self.config.PRE_NMS_LIMIT,
            nms_threshold=self.config.NMS_THRESHOLD
        )(rpn_merged)

        # MODEL
        if train:
            # Training model
            model = km.Model(inputs=[input_image, input_rpn_cls_gt, input_rpn_reg_deltas_gt, input_anchors],
                             outputs=[rpn_cls_loss, rpn_reg_loss],
                             **kwargs)
        else:
            # Inference Model: Leave out losses
            model = km.Model(inputs=[input_image, input_anchors],
                             outputs=[rpn_proposals, rpn_reg_deltas, rpn_cls],
                             **kwargs)
        return model

    def build_backbone_pretraining_model(self):
        input_image = kl.Input(shape=self.config.BACKBONE_TRAINING_IMAGE_SHAPE, name="backbone_training_input_image")
        backbone_output = self.build_backbone_model(input_image)
        output = kl.Dropout(rate=0.25)(backbone_output)
        output = kl.Flatten()(output)
        output = kl.Dense(units=64, activation="relu")(output)
        output = kl.Dropout(rate=0.5)(output)
        output = kl.Dense(units=self.config.NUM_BACKBONE_PRETRAINING_CLASSES, activation="softmax")(output)

        return km.Model(inputs=[input_image], outputs=[output])

    # def apply_rpn_reg_deltas(self, rpn_reg_deltas, anchors):
    #     """Apply rpn_reg_deltas encoded as described below to the anchor boxes and get rpn_reg.
    #
    #     Coordinate encoding ((x1,y1) = upper left corner, (x2, y2) = lower right corner):
    #     anchors = (x1=x_a-w_a/2, y1=y_a-h_a/2, x2=x_a+w_a/2, y2=y_a+h_a/2)
    #     rpn_reg = (x1=x-w/2, y1=y-h/2, x2=x+w/2, y2=y+w/2)
    #     rpn_reg_delta = ((x-x_a)/w_a, (y-y_a)/h_a, log(w/w_a), log(h/h_a))
    #
    #     :param np.array rpn_reg_deltas: [batch_size, num_anchors, 4: coord. deltas (dx, dy, dw, dh)]
    #         where dx, dy, dw, dh are as described above
    #     :param np.array anchors: [batch_size, num_anchors, 4] real anchor boxes
    #     :return: rpn_reg [batch_size, num_anchors, 4: (x1, y1, x2, y2)] predicted bounding boxes
    #     """
    #     # Convert to arrays of shape [batch_size, num_anchors]
    #     widths = anchors[:, :, 2] - anchors[:, :, 0]   # w_a
    #     heights = anchors[:, :, 3] - anchors[:, :, 1]  # h_a
    #     center_xs = anchors[:, :, 0] + 0.5 * widths    # x_a
    #     center_ys = anchors[:, :, 1] + 0.5 * heights   # y_a
    #     delta_xs = rpn_reg_deltas[:, :, 0]             # (x-x_a)/w_a
    #     delta_ys = rpn_reg_deltas[:, :, 1]             # (y-y_a)/h_a
    #     delta_widths = rpn_reg_deltas[:, :, 2]         # log(w/w_a)
    #     delta_heights = rpn_reg_deltas[:, :, 3]        # log(h/h_a)
    #
    #     # Apply deltas
    #     center_xs += delta_xs * widths
    #     center_ys += delta_ys * heights
    #     widths = widths * tf.exp(delta_widths)
    #     heights = heights * tf.exp(delta_heights)
    #
    #     # Convert back to x1s, y1s, x2s, y2s
    #     x1s = center_xs - 0.5 * widths
    #     y1s = center_ys - 0.5 * heights
    #     x2s = x1s + widths
    #     y2s = y1s + heights
    #
    #     # Concat back to box coordinates tensor
    #     # output_shape: [batch_size, num_anchors, 4]
    #     rpn_reg = tf.stack([x1s, y1s, x2s, y2s],
    #                        axis=2, name="apply_box_deltas_stack")
    #     return rpn_reg

    def get_anchors(self, batch_size):
        """Return anchors as tensor as needed for input."""
        anchors_shape = [batch_size, self.config.NUM_ANCHORS, 4]
        anchors_array = np.broadcast_to(self.config.ANCHOR_BOXES,
                                        anchors_shape)
        return anchors_array

    def compile_model(self):
        assert self.training_mode, "Can only compile a training-mode model."
        # OPTIMIZER
        # optimizer = keras.optimizers.SGD(
        #     lr=self.config.LEARNING_RATE,
        #     momentum=self.config.LEARNING_MOMENTUM,
        #     decay=self.config.WEIGHT_DECAY,
        #     clipnorm=self.config.GRADIENT_CLIP_NORM
        # )
        optimizer = keras.optimizers.Adadelta()

        # LOSSES
        # Clear losses
        self.model._losses = []
        self.model._per_input_losses = {}
        # In case weights should be applied (mind: the losses are already mean values)
        loss_layer_outputs = [self.model.get_layer(ln).output * self.config.LOSS_WEIGHTS[ln]
                              for ln in self.config.LOSS_LAYER_NAMES]
        self.model.add_loss(losses=loss_layer_outputs)

        self.model.compile(optimizer=optimizer,
                           loss=[None] * len(self.model.outputs))

        # METRICS
        # Add metrics for losses
        for loss_layer_name in self.config.LOSS_LAYER_NAMES:
            if loss_layer_name in self.model.metrics_names:
                continue

            self.model.metrics_names.append(loss_layer_name)

            loss_layer = self.model.get_layer(loss_layer_name)
            loss = (tf.reduce_mean(loss_layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS[loss_layer_name])
            self.model.metrics_tensors.append(loss)

    def compile_backbone_pretraining_model(self):
        self.backbone_pretraining_model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

    def predict(self, input_images):
        assert not self.training_mode
        batch_size = input_images.shape[0]
        anchors = self.get_anchors(batch_size)
        return self.model.predict(x=[input_images, anchors])

    @staticmethod
    def rpn_cls_loss_fn(rpn_cls_gt, rpn_cls_logits):
        """Log loss of anchor objectness classification.
        The ground-truth anchor objectness class is one of
            * 1: object
            * -1: no object
            * 0: neutral

        :param np.array rpn_cls_gt: [batch_size, num_anchors, 1] anchor ground-truth objectness class
        :param np.array rpn_cls_logits: [batch_size, num_anchors, 2: (bg, fg)] anchor bg, fg score logits
        :return: np.array [1,]: loss
        """
        # Squeeze last dim
        # output_shape: [batch_size, num_anchors: anchor classes]
        rpn_cls_gt = tf.squeeze(rpn_cls_gt, -1)

        # PICK NON-NEUTRAL ANCHORS
        # Only +1/-1 ground-truth anchors contribute to the loss,
        # neutral anchors (gt value = 0) don't.
        # output_shape: [batch_size, num_anchors: bool]
        non_neutral_indices = kb.not_equal(rpn_cls_gt, 0)
        # output_shape: [batch_size*num_non_neutral_anchors, 2: index (batch_idx, anchor_idx)]
        non_neutral_indices = tf.where(non_neutral_indices)
        rpn_class_logits = tf.gather_nd(rpn_cls_logits, non_neutral_indices)

        # CONVERT the +1/-1 ground-truth values to 0/1 values.
        rpn_cls_gt = kb.cast(kb.equal(rpn_cls_gt, 1), tf.int32)  # -1,0->0; 1->1
        rpn_cls_gt = tf.gather_nd(rpn_cls_gt, non_neutral_indices)

        # CROSSENTROPY LOSS
        loss = kb.sparse_categorical_crossentropy(target=rpn_cls_gt,
                                                  output=rpn_class_logits,
                                                  from_logits=True)
        # NORMALIZE BY NUMBER OF ANCHORS (=mean)
        loss = kb.switch(tf.size(loss) > 0, kb.mean(loss), tf.constant(0.0))
        return loss

    def rpn_reg_loss_fn(self, rpn_cls_gt, rpn_reg_deltas_gt, rpn_reg_deltas,
                        loss_fn=None):
        """Give the smooth L1-loss of all ground-truth positive anchors' coordinates.
        The ground-truth anchor objectness class is one of
            * 1: positive (contains object)
            * -1: negative (no object)
            * 0: neutral

        :param np.array rpn_cls_gt: [batch_size, num_anchors, 1]
            anchor ground-truth objectness class
        :param np.array rpn_reg_deltas_gt: [batch_size, num_anchors, 4: (x1, y1, x2, y2)]
            ground-truth bounding box coord;
            same order as rpn_reg, but padded with 0s for non-positive boxes
        :param np.array rpn_reg_deltas: [batch_size, num_anchors, 4: (x1, y1, x2, y2)]
            predicted bounding box coord
        :param function loss_fn: function to be applied to box coordinates
            (x, y) -> loss(x,y)
            default: smooth l1 loss;
        :return:
        """
        if loss_fn is None:
            loss_fn = self.loss_fn
        # PICK POSITIVE ANCHORS from rpn_reg
        # Only +1 ground-truth anchors contribute to the loss.
        # output_shape: [batch_size, num_anchors: objectness score]
        rpn_cls_gt = kb.squeeze(rpn_cls_gt, -1)
        # output_shape: [num_pos_anchors, 2: (batch idx, anchor idx)]
        positive_indices = tf.where(kb.equal(rpn_cls_gt, 1))
        # output_shape: [num_pos_anchors, 4: (x1, y1, x2, y2)]
        rpn_reg_deltas = tf.gather_nd(rpn_reg_deltas, positive_indices)

        # PICK POSITIVE ANCHORS from rpn_reg_gt
        # # Remove padding from rpn_reg_gt and flatten
        # # output: [batch_size, 1: num_pos_anchors for this batch]
        # batch_counts = kb.sum(kb.cast(kb.equal(rpn_cls_gt, 1), tf.int32), axis=1)
        # # output: [num_pos_anchors, 4: (x1, y1, x2, y2)]
        # rpn_reg_gt = MaskRCNNWrapper.batch_pack(rpn_reg_gt, batch_counts, self.config.BATCH_SIZE)
        rpn_reg_deltas_gt = tf.gather_nd(rpn_reg_deltas_gt, positive_indices)

        # Alternatively, if the number of sample boxes per batch varies much:
        # Do not normalize by taking mean, but by factor 1/self.config.NUM_ANCHORS
        #return loss_fn(rpn_reg_deltas, rpn_reg_deltas_gt, normalization_factor=1/self.config.NUM_ANCHORS)
        return loss_fn(rpn_reg_deltas, rpn_reg_deltas_gt)

    def build_rpn_model(self, model_input, anchors_per_location):
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
        shared = self.build_rpn_shared_model(model_input)

        # ANCHOR SCORES: cls model
        # Each anchor gets two filters, one for fg score, one for bg score
        # output_shape: [batch_size, anchors, 2]
        rpn_cls, rpn_cls_logits = MaskRCNNWrapper.build_rpn_cls_model(anchors_per_location, shared)

        # BOUNDING BOX REFINEMENT: reg model
        # Each anchor gets 4 filters, each for one part of the coordinate correction:
        # dx, dy, log(dw), log(dh)
        # output_shape: [batch_size, anchors, 4]
        rpn_reg = self.build_rpn_reg_model(anchors_per_location, shared)

        return rpn_cls, rpn_cls_logits, rpn_reg

    @staticmethod
    def build_rpn_reg_model(anchors_per_location, shared):
        """

        :param anchors_per_location:
        :param shared:
        :return: rpn_reg layer output with output_shape
            [batch_size, num_anchors, 4: (x,y,log(w),log(h))]
        """
        # Each anchor gets 4 filters, each for one part of the coordinate correction:
        # dx, dy, log(dw), log(dh)
        # output_shape: [batch_size, height, width, anchors_per_location*4: [dx,dy,log(dw),log(dh)]]
        bbox_refinement_output = kl.Conv2D(
            filters=anchors_per_location * 4,
            kernel_size=(1, 1),
            padding='same',
            activation='relu',
            name="rpn_reg_pred"
        )(shared)

        # Resize: row-wise concat anchors to list
        # output_shape: [batch_size, num_anchors, 4: (dx, dy, log(dw), log(dh))]
        rpn_reg = kl.Lambda(
            lambda t: tf.reshape(
                t, (
                    tf.shape(t)[0],
                    -1,
                    4)
            ),
            name="rpn_reg_resize"
        )(bbox_refinement_output)

        return rpn_reg

    @staticmethod
    def build_rpn_cls_model(anchors_per_location, model_input):
        """Model for getting anchor scores from rpn_shared_model.

        This yields an objectness score as (non-obj score, obj score) for each
        anchor based on the sliding window evaluated in the rpn_shared_model.
        """
        # Each anchor gets two filters, one for fg score, one for bg score
        # output_shape: [batch_size, height, width, anchors_per_location*2]
        anchor_scores_output = kl.Conv2D(
            filters=2 * anchors_per_location,
            kernel_size=(1, 1),
            padding='same',
            activation='linear',
            name="rpn_cls_raw"
        )(model_input)

        # Resize: row-wise concat anchors to list
        # output_shape: [batch_size, num_anchors, 2]
        rpn_cls_logits = kl.Lambda(
            lambda t: tf.reshape(
                t, (tf.shape(t)[0],  # batch_size
                    -1,  # rest
                    2)  # bg/fg score
            )
        )(anchor_scores_output)

        # Softmax: produce cls prediction
        # output_shape: as above
        rpn_cls = kl.Activation(
            'softmax', name="rpn_cls")(rpn_cls_logits)

        return rpn_cls, rpn_cls_logits

    def build_rpn_shared_model(self, model_input):
        """Output of shared ConvLayer for region proposal parts.

        This emulates the sliding window which is evaluated by the coordinate correction
        estimator and the objectness class estimator.
        Thus choose the kernel_size (=sliding window size) such that a typical object fits
        inside the kernel window."""
        shared_output = kl.Conv2D(
            filters=32,
            kernel_size=self.config.SLIDING_WINDOW_SIZE,
            # exclude sliding windows that would cross the image borders:
            padding='valid',
            activation='relu',
            name="rpn_conv_shared"
        )(model_input)
        return shared_output

    @staticmethod
    def build_backbone_model(image_input):
        """Build convolutional backbone model.

        Mind to set self.config.DOWNSCALING_FACTOR to this model's downscaling rate.

        :param tuple image_input: keras.layers.Input(); shape should be divisible by 2**3
        :return: backbone_model as layer output
        """
        # CONV1: 5x5 Conv, MaxPooling
        out = kl.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',  # same takes ridiculously much longer than valid...
            activation='relu',
            name="backbone_conv1"
        )(image_input)
        out = kl.MaxPooling2D(pool_size=(2, 2), padding='same', name="backbone_maxpool1")(out)

        # CONV2: 3x3 Conv, MaxPooling
        out = kl.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            name="backbone_conv2"
        )(out)
        out = kl.MaxPooling2D(pool_size=(2, 2), padding='same', name="backbone_maxpool2")(out)

        # CONV3: 3x3 Conv, 3x3 Conv, MaxPooling
        # out = kl.Conv2D(
        #     filters=16,
        #     kernel_size=(3, 3),
        #     padding='same',
        #     activation='relu',
        #     name="backbone_conv3_1"
        # )(out)
        out = kl.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            name="backbone_conv3_2"
        )(out)
        out = kl.MaxPooling2D(pool_size=(2, 2), padding='same', name="backbone_maxpool3")(out)

        # CONV4: 3x3 Conv, 3x3 Conv, MaxPooling
        # out = kl.Conv2D(
        #     filters=16,
        #     kernel_size=(3, 3),
        #     padding='same',
        #     activation='relu',
        #     name="backbone_conv4_1"
        # )(out)
        out = kl.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            name="backbone_conv4_2"
        )(out)
        out = kl.MaxPooling2D(pool_size=(2, 2), padding='same', name="backbone_maxpool4")(out)

        # optional Dropout
        # out = kl.Dropout(rate=0.25, name="backbone_dropout")(out)

        # backbone_model = km.Model(input=image_input, output=backbone_output)
        return out

    def fit_model(self,
                  inputs,
                  outputs=None,
                  validation_split=None,
                  checkpt_filepath_format=None):
        """Fit self.model.

        :param list inputs:
        [
            nd.array input_image config.IMAGE_SHAPE: [batch_size, height, width, channels]
            nd.array input_rpn_cls_gt config.RPN_CLS_SHAPE: [batch_size, num_proposals, 1],
            nd.array input_rpn_reg_gt config.RPN_REG_SHAPE: [batch_size, num_proposals, 4: (dx, dy, log(dw), log(dh))]
        ]
        :param list outputs: leave empty due to custom losses; compare ground-truth part of input
        [
            rpn_cls_loss,  # do not consider!
            rpn_reg_loss  # do not consider!
        ]
        :param float validation_split: percentage of the input to take for validation
        :param str checkpt_filepath_format: format string for the path to the model weights checkpointing file;
        accepts parameters like epoch, see keras documentation for checkpointing callback
        """
        if outputs is None:
            outputs = []
        validation_split = validation_split or self.config.VALIDATION_SPLIT

        num_samples = inputs[0].shape[0]
        anchors = self.get_anchors(batch_size=num_samples)
        inputs.append(anchors)

        # Stop before overfitting and checkpoint every epoch
        callbacks = [kc.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=1)]
        if checkpt_filepath_format is not None:
            callbacks.append(
                kc.ModelCheckpoint(checkpt_filepath_format, monitor='val_loss', save_weights_only=True, period=1)
            )
        self.model.fit(
            x=inputs,
            y=outputs,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            verbose=1,
            validation_split=validation_split,
            callbacks=callbacks
        )

    def fit_backbone_training_model(self, inputs, labels, validation_data=None, validation_split=None):
        kwargs = {}
        if validation_data is None:
            kwargs["validation_split"] = validation_split or self.config.VALIDATION_SPLIT
        else:
            kwargs["validation_data"] = validation_data
        callbacks = [
            kc.EarlyStopping(monitor='val_acc', min_delta=0.015, patience=1),
            kc.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=1)
            ]
        self.backbone_pretraining_model.fit(
            x=inputs, y=labels,
            batch_size=self.config.BACKBONE_PRETRAINING_BATCH_SIZE,
            epochs=self.config.BACKBONE_PRETRAINING_EPOCHS,
            verbose=1,
            callbacks=callbacks,
            **kwargs)


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
                 config=None,
                 **kwargs):
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

        :param tensor inputs: np.array of shape [batch_size, num_anchors, 2+4+4: rpn_cls+rpn_reg+anchors]
            where the data consists of np.arrays (all coordinates normalized):
            - *rpn_cls*: 2 scores (bg prob, fg prob)
            - *rpn_reg*: 4 coord. corrections (dx, dy, log(dw), log(dh))
            - *anchors*: 4 coord. (x1, y1, x2, y2)
        :return: Tensor of proposals in normalized coordinates
            of shape [batch_size, num_proposals, (x1, y1, x2, y2)]
            (possibly padded with 0s)
        """
        # Iterate over every datum in batch
        # proposals = []
        # for i in range(0, self.batch_size):
        #     datum = inputs[i]
        #     proposals.append(self.get_proposals(datum))
        # return tf.stack(proposals, name="stack_proposals")
        return tf.map_fn(self.get_proposals, inputs)

    def get_proposals(self, datum):
        # TRIMMING
        # Improve performance by trimming to max. pre_nms_limit number of best anchors
        # (sorted by fg score) and doing the rest on the smaller subset of size N.
        # output_shape: ([N, 2 (scores) + 4 (coord corr) + 4 (coord)])
        datum = self.trim_to_top_anchors(datum)
        # VARIABLES
        fg_scores = datum[:, 1]  # Foreground class confidence of shape [N,]
        rpn_reg_deltas = datum[:, 2:6]  # Box coordinate correction deltas of shape [N, 4]
        anchors = datum[:, 6:10]  # Anchor box coordinates of shape [N, 4]
        # REFINE ANCHORS
        # Apply coordinate_deltas to anchors to get refined anchor boxes.
        # output_shape: [N, 4: (x1, y1, x2, y2)]
        boxes = self.apply_box_deltas(anchors, rpn_reg_deltas)
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
        # output_shape: [self.num_proposals, 4: coord]
        datum_proposals = self.pad_proposals(datum_proposals)
        return datum_proposals

    def trim_to_top_anchors(self, datum):
        """Trim anchors for datum to the top self.pre_nms_limit number of anchors,
        measured by foreground score.

        :param np.array datum: of shape
            [
                num_anchors,
                2 scores (bg, fg) +
                4 coord corr (dx, dy, log(dw), log(dh) +
                4 coord (x1, y1, x2, y2)
            ]
            )
        """

        anchor_axis = 0
        num_anchors = tf.shape(datum)[0]
        fg_scores = datum[:, 1]
        pre_nms_limit = tf.minimum(self.pre_nms_limit, num_anchors)

        # top k anchor indices
        indices_to_keep = tf.nn.top_k(fg_scores,
                                      k=pre_nms_limit,
                                      sorted=True, name="top_anchors").indices
        # drop very uncertain ones
        indices_to_keep = tf.gather(indices_to_keep, tf.where(tf.gather(fg_scores, indices_to_keep) > 0.5)[:, 0])

        # trim the datum to the above top k anchors
        return tf.gather(datum,
                         indices=indices_to_keep,
                         axis=anchor_axis)

    @staticmethod
    def apply_box_deltas(boxes, deltas):
        """Applies the given deltas to the given boxes.

        * box center point, width, height:
            (x_a, y_a, w_a, h_a)
        * box coordinates:
            (x1_a = x_a-w_a/2, y1_a = y_a-h_a/2, x2_a = x_a+w_a/2, y2_a = y_a+h_a/2)
        * box_delta:
            (dx = (x-x_a)/w_a, dy = (y-y_a)/h_a, log(dw=w/w_a), log(dh=h/h_a))
        * corrected box center point, width, height:
            (x = x_a + dx*w_a, y = y_a + dy*h_a, w_a*exp(dw), h_a*exp(dh))
        * corrected box coordinates:
            (x1 = x-w/2, y1 = y-h/2, x2 = x+w/2, y2 = y+w/2)

        All coordinates normalized.
        :param np.array boxes: [N, 4: (x1_a, y1_a, x2_a, y2_a)] boxes to update with deltas
        :param np.array deltas: [N, 4: (dx, dy, log(dw), log(dh))] refinements to apply
        :return: np.array [N, 4: (x1, y1, x2, y2)] updated box coordinates
        """
        # Delta values
        dx, dy = deltas[:, 0], deltas[:, 1]
        dw, dh = tf.exp(deltas[:, 2]), tf.exp(deltas[:, 3])

        # Convert to (center_x, center_y, w, h) coordinate scheme, each component of shape [N,]
        x1_a, y1_a, x2_a, y2_a = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w_a = x2_a - x1_a
        h_a = y2_a - y1_a
        center_x_a = x1_a + 0.5 * w_a
        center_y_a = y1_a + 0.5 * h_a

        # Apply deltas
        center_x = center_x_a + dx * w_a
        center_y = center_y_a + dy * h_a
        w = w_a * dw
        h = h_a * dh

        # Convert to x1, y1, x2, y2
        x1 = center_x - 0.5 * w
        y1 = center_y - 0.5 * h
        x2 = x1 + w
        y2 = y1 + h

        # Concat back to box coordinates tensor
        # output_shape: [N, 4]
        result_coordinates = tf.stack([x1, y1, x2, y2],
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
        # output_shape: 4 x [N, 1]
        x1s, y1s, x2s, y2s = tf.split(boxes, 4, axis=1)
        win_x1s, win_y1s, win_x2s, win_y2s = tf.split(window, 4)

        # Clip
        x1s = tf.maximum(tf.minimum(x1s, win_x2s), win_x1s)  # overlap right: right border, overlap left: left border
        y1s = tf.maximum(tf.minimum(y1s, win_y2s), win_y1s)
        x2s = tf.maximum(tf.minimum(x2s, win_x2s), win_x1s)
        y2s = tf.maximum(tf.minimum(y2s, win_y2s), win_y1s)
        # output_shape: [N, 4]
        clipped_coordinates = tf.stack([x1s, y1s, x2s, y2s],
                                       axis=1, name="clipped_boxes")
        clipped_coordinates = tf.squeeze(clipped_coordinates, -1)
        return clipped_coordinates

    def pad_proposals(self, proposals):
        """Pad proposals with 0s until they have length self.num_proposals.

        :param np.array proposals: [n, 4] list of proposed box coordinates
        :return: np.array [self.num_proposals, 4], proposals padded with 0s at the end
        """
        curr_num_proposals = tf.shape(proposals)[0]
        padding = tf.maximum(self.num_proposals - curr_num_proposals, 0)
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
        return proposals
