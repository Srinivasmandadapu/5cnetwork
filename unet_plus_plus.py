from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def unet_plus_plus(input_shape=(128, 128, 3)):
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    # Decoder
    u6 = UpSampling2D((2, 2))(p1)
    merge6 = concatenate([u6, c1], axis=3)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c6)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
