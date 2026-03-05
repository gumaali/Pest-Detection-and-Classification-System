from keras import layers, models
import numpy as np
from keras.src.optimizers import Adam
from Classificaltion_Evaluation import net_evaluation


def Model_CNN_Seg(Data, Target, SPE=None, HN=None, BS=None, sol=None):
    if SPE is None:
        SPE = 5
    if HN is None:
        HN = 128
    if BS is None:
        BS = 32

    IMG_SIZE = 32
    NUM_CLASSES = 3
    X = np.zeros((Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Data.shape[0]):
        temp = np.resize(Data[i], (IMG_SIZE * IMG_SIZE, 3))
        X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Y = np.zeros((Target.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Target.shape[0]):
        temp = np.resize(Target[i], (IMG_SIZE * IMG_SIZE, 3))
        Y[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    # Define segmentation model (Encoder-Decoder structure)
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    outputs = layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(X, Y, epochs=10, steps_per_epoch=SPE, batch_size=BS, validation_split=0.25)
    pred = model.predict(X)
    Eval = [net_evaluation(X[n].astype('uint8'), pred[n].astype('uint8')) for n in range(pred.shape[0])]
    EVAl = np.mean(Eval, axis=0)
    return pred, EVAl

