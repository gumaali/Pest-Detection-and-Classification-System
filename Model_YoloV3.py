import keras
from keras.src.models import Model
from keras.src.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Add, UpSampling2D, Concatenate
import numpy as np
from Classificaltion_Evaluation import net_evaluation


#  Define Darknet-53 Backbone (used by YOLOv3)
def conv_block(x, filters, kernel_size, strides=1):
    """Conv → BN → LeakyReLU"""
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def residual_block(x, filters, blocks):
    """Residual block group"""
    for _ in range(blocks):
        y = conv_block(x, filters // 2, 1)
        y = conv_block(y, filters, 3)
        x = Add()([x, y])
    return x


def darknet53(input_tensor):
    """Darknet-53 backbone"""
    x = conv_block(input_tensor, 32, 3)
    x = conv_block(x, 64, 3, strides=2)
    x = residual_block(x, 64, 1)

    x = conv_block(x, 128, 3, strides=2)
    x = residual_block(x, 128, 2)
    skip_1 = x

    x = conv_block(x, 256, 3, strides=2)
    x = residual_block(x, 256, 8)
    skip_2 = x

    x = conv_block(x, 512, 3, strides=2)
    x = residual_block(x, 512, 8)
    skip_3 = x

    x = conv_block(x, 1024, 3, strides=2)
    x = residual_block(x, 1024, 4)
    return skip_1, skip_2, skip_3, x


# YOLOv3 Detection Head
def yolo_head(x, num_classes):
    x = conv_block(x, 512, 1)
    x = conv_block(x, 1024, 3)
    x = conv_block(x, 512, 1)
    x = conv_block(x, 1024, 3)
    x = conv_block(x, 512, 1)
    output = Conv2D(3 * (num_classes + 5), 1, padding='same')(x)  # 3 anchors per scale
    return output


#  Full YOLOv3 Model
def yolo_v3_model(input_size=(256, 256, 3), num_classes=3):
    inputs = Input(input_size)
    skip_1, skip_2, skip_3, x = darknet53(inputs)

    # Detection head for large objects
    y1 = yolo_head(x, num_classes)

    # Upsample + concat for medium objects
    x = conv_block(x, 256, 1)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_3])
    y2 = yolo_head(x, num_classes)

    # Upsample + concat for small objects
    x = conv_block(x, 128, 1)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_2])
    y3 = yolo_head(x, num_classes)

    model = Model(inputs, [y1, y2, y3])
    return model


def Model_YoloV3(Images, GT, HN=None, sol=None):
    if sol is None:
        sol = [4, 50, 0, 5]
    if HN is None:
        HN = 64

    IMG_SIZE = 256
    classes = 3
    optimizer = ['SGD', 'Adam', 'RMSprop', 'Adagrad', 'Adadelta']
    input_shape = (IMG_SIZE, IMG_SIZE, 3)

    # Resize input data
    Train_Temp = np.zeros((Images.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(Images.shape[0]):
        Train_Temp[i, :] = np.resize(Images[i], (input_shape[0], input_shape[1], input_shape[2]))
    Train_X = Train_Temp.reshape(Train_Temp.shape[0], input_shape[0], input_shape[1], input_shape[2])

    Test_Temp = np.zeros((GT.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(GT.shape[0]):
        Test_Temp[i, :] = np.resize(GT[i], (input_shape[0], input_shape[1], input_shape[2]))
    Train_Y = Test_Temp.reshape(Test_Temp.shape[0], input_shape[0], input_shape[1], input_shape[2])

    # Build YOLOv3
    model = yolo_v3_model(input_size=input_shape, num_classes=classes)
    model.compile(optimizer=optimizer[int(sol[2])], loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(Train_X, [Train_Y, Train_Y, Train_Y], epochs=sol[1], steps_per_epoch=2, verbose="auto")
    Predict = model.predict(Train_X)
    Eval = [net_evaluation(Train_Y[n].astype('uint8'), Predict[n].astype('uint8')) for n in range(Predict.shape[0])]
    EVAl = np.mean(Eval, axis=0)
    return Predict, EVAl
