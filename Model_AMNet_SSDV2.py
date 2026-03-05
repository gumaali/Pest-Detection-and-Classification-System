from keras import layers, models
import numpy as np
from keras.src.applications.mobilenet_v2 import MobileNetV2
from keras.src.optimizers import Adam
from Classificaltion_Evaluation import net_evaluation


def Model_AMNet_SSDV2(Data, Target, SPE=None, HN=None, BS=None, sol=None):

    if SPE is None:
        SPE = 5
    if HN is None:
        HN = 128
    if BS is None:
        BS = 32

    IMG_SIZE = 32
    NUM_CLASSES = 3

    # Data Reshaping
    X = np.zeros((Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Data.shape[0]):
        temp = np.resize(Data[i], (IMG_SIZE * IMG_SIZE, 3))
        X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Y = np.zeros((Target.shape[0], NUM_CLASSES))
    for i in range(Target.shape[0]):
        Y[i] = Target[i]

    # Backbone (MobileNetV2)
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None
    )

    x = base_model.output

    # Adaptive Feature Layers
    x = layers.DepthwiseConv2D(3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, (1,1), activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.DepthwiseConv2D(3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (1,1), activation='relu')(x)

    # SSD Detection Head
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(HN, activation='relu')(x)

    bbox_output = layers.Dense(4, activation='linear', name='bbox')(x)
    class_output = layers.Dense(NUM_CLASSES, activation='softmax', name='class')(x)

    outputs = layers.Concatenate()([bbox_output, class_output])

    model = models.Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    model.fit(
        X, Y,
        epochs=10,
        steps_per_epoch=SPE,
        batch_size=BS,
        validation_split=0.25
    )
    pred = model.predict(X)
    Eval = [net_evaluation(X[n].astype('uint8'), pred[n].astype('uint8')) for n in range(pred.shape[0])]
    EVAl = np.mean(Eval, axis=0)
    return pred, EVAl