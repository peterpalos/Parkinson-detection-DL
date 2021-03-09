from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Attention
from tensorflow.keras.layers import Dropout

#from tensorflow.keras.utils import plot_model

def create_mlp():
    input = Input(shape=(20,))
    x = Dense(64, activation='relu')(input)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(64, activation='relu')(x)

    model = Model(inputs=input, outputs=output)
    return model

def create_resnet():
    resnet = ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    for layer in resnet.layers[0:123]:
        layer.trainable = False
    output = Flatten()(resnet.output)

    model = Model(inputs=resnet.input, outputs=output)
    return model

def create_CNN_LSTM():
    input = Input(shape=(224, 224, 1))
    x = Conv2D(filters=64, kernel_size=2, strides=1, padding="same")(input)
    x = MaxPooling2D()(x)
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Bidirectional(LSTM(units=128, dropout=0.3, recurrent_dropout=0.2))(x)
    output = Attention(dropout=0.3)(x)

    model = Model(inputs=input, outputs=output)
    return model


def create_mixed_model(resnet=True):
    mlp = create_mlp()
    if resnet:
        resnet = create_resnet()
        combined_model = concatenate([mlp.output, resnet.output])
    else:
        cnn_lstm =create_CNN_LSTM()
        combined_model = concatenate([mlp.output, cnn_lstm.output])

    x = Dense(64, activation="relu")(combined_model)
    x = Dense(248, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)


    model = Model(inputs=[mlp.input, resnet.input], outputs=x)
    return model

