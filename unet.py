from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def unet():

 x_in = Input(shape=(512, 512, 3))

 x = Conv2D(32, (3, 3), activation='relu', padding='same')(x_in)
 x = BatchNormalization()(x)
 x = Dropout(0.2)(x)
 x_skip1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
 x = MaxPooling2D((2, 2))(x_skip1)

 x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
 x = BatchNormalization()(x)
 x = Dropout(0.2)(x)
 x_skip2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
 x = MaxPooling2D((2, 2))(x_skip2)

 x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
 x = BatchNormalization()(x)
 x = Dropout(0.2)(x)
 x_skip3 = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
 x = MaxPooling2D((2, 2))(x_skip3)

 x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
 x = BatchNormalization()(x)
 x = Dropout(0.2)(x)
 x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
 x = BatchNormalization()(x)

 x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
 x = BatchNormalization()(x)
 x = Dropout(0.2)(x)
 x = Conv2DTranspose(256, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
 x = BatchNormalization()(x)
 x = Concatenate()([x, x_skip3])

 x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
 x = BatchNormalization()(x)
 x = Dropout(0.2)(x)
 x = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
 x = BatchNormalization()(x)
 x = Concatenate()([x, x_skip2])

 x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
 x = BatchNormalization()(x)
 x = Dropout(0.2)(x)
 x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
 x = BatchNormalization()(x)
 x = Concatenate()([x, x_skip1])

 x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
 x = BatchNormalization()(x)
 x = Dropout(0.2)(x)
 x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
 x = BatchNormalization()(x)

 x_out = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

 model = Model(x_in,x_out)

 model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

 return model
 