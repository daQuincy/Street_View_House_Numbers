from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input

from keras.models import Model

class SVHNNet:
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        
        inputs = Input(shape=inputShape)
        
        # layer 1
        x = Conv2D(filters=48, kernel_size=5, padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=2, strides=2, padding="same")(x)
        x = Dropout(0.25)(x)
        
        # layer 2
        x = Conv2D(filters=64, kernel_size=5, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=2, strides=1, padding="same")(x)
        x = Dropout(0.25)(x)
        
        # layer 3
        x = Conv2D(filters=128, kernel_size=5, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=2, strides=2, padding="same")(x)
        x = Dropout(0.25)(x)
        
        # layer 4
        x = Conv2D(filters=160, kernel_size=5, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=2, strides=1, padding="same")(x)
        x = Dropout(0.25)(x)
        
        # layer 5
        x = Conv2D(filters=192, kernel_size=5, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=2, strides=2, padding="same")(x)
        x = Dropout(0.25)(x)
        
        # layer 6
        x = Conv2D(filters=192, kernel_size=5, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=2, strides=1, padding="same")(x)
        x = Dropout(0.25)(x)
        
        # layer 7
        x = Conv2D(filters=192, kernel_size=5, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=2, strides=2, padding="same")(x)
        x = Dropout(0.25)(x)
        
        # layer 8
        x = Conv2D(filters=192, kernel_size=5, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=2, strides=1, padding="same")(x)
        x = Dropout(0.25)(x)
        
        # layer 9
        x = Flatten()(x)
        x = Dense(units=3072)(x)
        x = Activation("relu")(x)
        x = Dropout(0.25)(x)
        
        # layer 10
        x = Dense(units=3072)(x)
        x = Activation("relu")(x)
        x = Dropout(0.25)(x)
        
        # prediction layers
        digit1 = Dense(classes, activation="softmax")(x)
        digit2 = Dense(classes, activation="softmax")(x)
        digit3 = Dense(classes, activation="softmax")(x)
        digit4 = Dense(classes, activation="softmax")(x)
        digit5 = Dense(classes, activation="softmax")(x)
        seq_len = Dense(6, activation="softmax")(x)
        
        model = Model(inputs=inputs, outputs=[digit1, digit2, digit3, digit4, digit5, seq_len])
        
        return model
        
        