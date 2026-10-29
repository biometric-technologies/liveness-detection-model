from keras.applications import mobilenet_v2
from keras.layers import Conv2D, Dropout, Dense
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.metrics import AUC


def GetModel(img_width, img_height):
    pretrain_net = mobilenet_v2.MobileNetV2(input_shape=(img_width, img_height, 3), include_top=False, weights='imagenet')

    for layer in pretrain_net.layers:
        layer.trainable = True

    x = pretrain_net.output
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(rate=0.2)(x)

    x = Dense(1, activation='sigmoid', name='classifier')(x)

    model = Model(inputs=pretrain_net.input, outputs=x, name='mobilenetv2_spoof')
    learning_rate = 0.0001
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['acc', AUC(name='auc')])

    return model
