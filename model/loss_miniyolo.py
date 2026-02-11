"""This file contains the YOLOv1 loss implementation, with its relative IOU and box manipulation decoder functions."""

import tensorflow as tf


class MiniyoloLoss(tf.keras.losses.Loss):
    """Defines the MiniYOLO loss function overriding the call functions as in Keras documentation."""

    def __init__(self, S, B, C, lambda_coord=5.0, lambda_noobj=0.5):
        """Initializes the loss class with the YOLOv1 relative parameters.

        Args:
            S (int): Number of division of the image (S² is the total number of cells). Ex. S=2 we divide the image in cells pixel wise ([0,0],[0,1],[1,0],[1,1]), 4 cells total.
            B (int): Maximum number of boxes recognizable by the model for each cell. Only one is actually filled for each image in the dataset.
            C (int): Number of classes recognizable by the model.
            lambda_coord (float): Hyperparameter that increases the loss from bounding box coordinate predictions.
            lambda_noobj (float): Hyperparameter that decreases the loss from confidence predictions for boxes that don't contain objects.
        """

        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def call(self, y_true, y_pred):
        """Calculates the batch normalized loss for the YOLOv1 implementation.

        Args:
            y_true (Tensor): (Batch, S, S, B*5 + C) true target tensor.
            y_pred (Tensor): (Batch, S, S, B*5 + C) model predicted target tensor to compare against the y_true target.

        Returns:
            float: Float32 scalar tensor containing the total loss for the model, normalized on the batch size.
        """

        batch_size = tf.cast(tf.shape(y_pred)[0], tf.float32)

        # Separate predicted boxes and labels data for predicted and true target tensors

        # (batch, S, S, B, 5) after reshape
        pred_boxes = tf.reshape(
            y_pred[..., : self.B * 5],
            (-1, self.S, self.S, self.B, 5),
        )

        # (batch,S,S,C)
        pred_classes = y_pred[..., self.B * 5 :]

        # (batch, S, S, B, 5) after reshape
        true_boxes = tf.reshape(
            y_true[..., : self.B * 5],
            (-1, self.S, self.S, self.B, 5),
        )

        # (batch,S,S,C)
        true_classes = y_true[..., self.B * 5 :]

        obj_mask_box = true_boxes[..., 4:5]
        obj_mask_cell = tf.reduce_max(obj_mask_box, axis=3)  # (B,S,S,1)

        # Use the single ground truth box in the cell (assumes only box 0 is filled in targets)
        true_boxes_single = true_boxes[..., 0:1, 0:4]  # (B,S,S,1,4)

        # Decode boxes
        pred_xyxy = self._decode_boxes(pred_boxes)
        true_xyxy = self._decode_boxes(true_boxes_single)

        # IoU against single ground truth box for all B predictions
        ious = self._iou(pred_xyxy, true_xyxy)  # (B,S,S,B)
        best_iou = tf.reduce_max(ious, axis=3, keepdims=True)

        # Pick a single responsible box
        best_idx = tf.argmax(ious, axis=3)  # (B,S,S)
        responsible = tf.one_hot(best_idx, depth=self.B, dtype=tf.float32)
        responsible = responsible[..., None]  # (B,S,S,B,1)
        responsible = responsible * obj_mask_cell[..., None]

        # Coordinate loss
        true_xywh = tf.tile(true_boxes_single, [1, 1, 1, self.B, 1])
        pred_xy = pred_boxes[..., 0:2]
        true_xy = true_xywh[..., 0:2]
        pred_wh = tf.maximum(pred_boxes[..., 2:4], 1e-6)
        true_wh = tf.maximum(true_xywh[..., 2:4], 1e-6)

        pred_wh_sqrt = tf.sqrt(pred_wh)
        true_wh_sqrt = tf.sqrt(true_wh)

        coord_loss = tf.reduce_sum(
            responsible
            * (tf.square(pred_xy - true_xy) + tf.square(pred_wh_sqrt - true_wh_sqrt))
        )

        # Confidence loss (obj) uses best IoU target
        conf_obj_loss = tf.reduce_sum(
            responsible * tf.square(pred_boxes[..., 4:5] - best_iou[..., None])
        )

        # Confidence loss (no-obj in empty cells + non-responsible in object cells)
        noobj_mask = (1.0 - obj_mask_cell)[..., None] + obj_mask_cell[..., None] * (
            1.0 - responsible
        )
        conf_noobj_loss = tf.reduce_sum(noobj_mask * tf.square(pred_boxes[..., 4:5]))

        # Class loss
        class_loss = tf.reduce_sum(
            obj_mask_cell * tf.square(pred_classes - true_classes)
        )

        # Total loss
        total_loss = (
            self.lambda_coord * coord_loss
            + conf_obj_loss
            + self.lambda_noobj * conf_noobj_loss
            + class_loss
        )

        # Debug only
        # tf.print(
        #     " - COORD Loss: ",
        #     coord_loss,
        #     " - CONF OBJ Loss: ",
        #     conf_obj_loss,
        #     " - CONF NOOBJ Loss: ",
        #     conf_noobj_loss,
        #     " - CLASS Loss: ",
        #     class_loss,
        # )

        return total_loss / batch_size

    def _decode_boxes(self, boxes):
        """Converts the x_offset, y_offset, width and height into x1, y1, x2, y2 corner position relative to the image.

        Args:
            boxes (Tensor): Target tensor relative data to the boxes, so from the model output and true target values (Batch, S, S, B*5 + C), it takes in (
            Batch, S, S, B*5) containing (Batch, S, S, B*(x_offset, y_offset, width, height, confidence)) (confidence will not be used).

        Returns:
            Tensor: x1, y1, x2, y2 corner position relative to the image tensor for each B boxes.
        """

        grid_x = tf.range(self.S, dtype=tf.float32)
        grid_y = tf.range(self.S, dtype=tf.float32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)

        grid_x = tf.reshape(grid_x, (1, self.S, self.S, 1))
        grid_y = tf.reshape(grid_y, (1, self.S, self.S, 1))

        x = (grid_x + boxes[..., 0]) / self.S
        y = (grid_y + boxes[..., 1]) / self.S
        w = boxes[..., 2]
        h = boxes[..., 3]

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        return tf.stack([x1, y1, x2, y2], axis=-1)

    def _iou(self, box1, box2):
        """Calculates the Intersection Over Union of 2 given boxes after the boxes are decoded to a x1, y1, x2, y2 format, which outputs a
        tensor containing how close the boxes match between them. IOU function divides the area of intersection between the 2 boxes
        with the union of their area.

        Args:
            box1 (Tensor): First box in x1, y1, x2 and y2 format which represent the normalized corner of the box relative to the image.
            box2 (Tensor): Second box in x1, y1, x2 and y2 format which represent the normalized corner of the box relative to the image.

        Returns:
            Result Tensor: Tensor containing the calculated IOU, 1e-6 used to avoid division by 0.
        """

        x1 = tf.maximum(box1[..., 0], box2[..., 0])
        y1 = tf.maximum(box1[..., 1], box2[..., 1])
        x2 = tf.minimum(box1[..., 2], box2[..., 2])
        y2 = tf.minimum(box1[..., 3], box2[..., 3])

        inter = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)

        area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

        union = area1 + area2 - inter

        return inter / (union + 1e-6)
