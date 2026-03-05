import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from Classificaltion_Evaluation import net_evaluation


# Backbone Network
def backbone(input_tensor):
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    return x


# Region Proposal Network (RPN)
def rpn(feature_map, num_anchors=9):
    shared = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(feature_map)

    # Objectness score
    rpn_cls = layers.Conv2D(num_anchors, (1, 1), activation='sigmoid', name='rpn_cls')(shared)

    # Bounding box regression
    rpn_reg = layers.Conv2D(num_anchors * 4, (1, 1), activation='linear', name='rpn_reg')(shared)

    return rpn_cls, rpn_reg


# ROI Pooling Layer
class ROIPooling(layers.Layer):
    def __init__(self, pool_size, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs):
        feature_map, rois = inputs
        pooled_rois = []

        for roi in rois:
            x, y, w, h = tf.cast(roi, tf.int32)
            roi_feature = feature_map[:, y:y + h, x:x + w, :]
            pooled = tf.image.resize(roi_feature, (self.pool_size, self.pool_size))
            pooled_rois.append(pooled)

        return tf.concat(pooled_rois, axis=0)


def classifier_head(roi_features, num_classes):
    x = layers.Flatten()(roi_features)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)

    cls_output = layers.Dense(num_classes, activation='softmax', name='cls_output')(x)
    reg_output = layers.Dense(num_classes * 4, activation='linear', name='bbox_output')(x)
    return cls_output, reg_output


def Model_Faster_RCNN(Data, Target, ACT=None):
    if ACT is None:
        ACT = 'relu'
    IMG_SIZE = 32
    NUM_CLASSES = 3
    # Prepare input data
    X = np.zeros((Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Data.shape[0]):
        temp = np.resize(Data[i], (IMG_SIZE * IMG_SIZE, 3))
        X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    # Prepare targets (one-hot)
    Y = np.zeros((Target.shape[0], NUM_CLASSES))
    for i in range(Target.shape[0]):
        Y[i] = Target[i]

    # Build the model
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Backbone
    x = layers.Conv2D(64, (3, 3), activation=ACT, padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    # Flatten + Dense Head
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X, Y, epochs=10, batch_size=32, steps_per_epoch=5, validation_split=0.25)
    pred = model.predict(X)
    Eval = [net_evaluation(Y[n].astype('uint8'), pred[n].astype('uint8')) for n in range(len(X))]
    EVAl = np.mean(Eval, axis=0)
    return pred, EVAl
