import numpy as np
from keras.src.models import Model
from keras.src.layers import Input, Conv2D, BatchNormalization, Activation
from keras.src.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply
from keras.src.layers import Add, Concatenate
from keras.src.optimizers import Adam
from Classificaltion_Evaluation import ClassificationEvaluation


# Squeeze and Excitation Block
def SE_Block(x, ratio=16):
    filters = x.shape[-1]

    se = GlobalAveragePooling2D()(x)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)

    x = Multiply()([x, se])
    return x


# -----------------------------
# Residual Block
# -----------------------------
def Residual_Block(x, filters):

    shortcut = x

    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)

    x = SE_Block(x)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


# Dense Residual Block
def Dense_Res_Block(x, filters):

    r1 = Residual_Block(x, filters)
    c1 = Concatenate()([x, r1])

    r2 = Residual_Block(c1, filters)
    c2 = Concatenate()([c1, r2])

    r3 = Residual_Block(c2, filters)
    out = Concatenate()([c2, r3])

    return out


# DRNet-SE Model
def Model_DRNet_SE(train_data, train_target, test_data, test_target, EP=None, BS=None, HN=None, sol=None):

    if sol is None:
        sol = [1]
    if BS is None:
        BS = 4
    if HN is None:
        HN = 64
    if EP is None:
        EP = 2

    Classes = test_target.shape[-1]
    IMG_SIZE = 32

    # Reshape data
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    # Model Architecture
    inputs = Input(shape=(32,32,3))

    x = Conv2D(64, (3,3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense_Res_Block(x, 64)
    x = Dense_Res_Block(x, 64)

    x = GlobalAveragePooling2D()(x)

    x = Dense(HN, activation='relu')(x)
    outputs = Dense(Classes, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        Train_X, train_target,
        epochs=EP,
        batch_size=BS,
        validation_data=(Test_X, test_target)
    )
    pred = model.predict(Test_X)
    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = np.asarray(pred).astype('int')
    Eval = ClassificationEvaluation(test_target, pred)
    return Eval, pred