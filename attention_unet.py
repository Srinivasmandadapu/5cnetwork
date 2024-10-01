from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Activation
from tensorflow.keras.layers import add, multiply

def attention_gate(x, g, inter_channel):
    # Apply a 1x1 convolution to both x and g without changing spatial dimensions
    theta_x = Conv2D(inter_channel, (1, 1), padding='same')(x)  # No stride, so same spatial dimensions
    phi_g = Conv2D(inter_channel, (1, 1), padding='same')(g)    # Gating signal
    add_xg = add([theta_x, phi_g])  # Element-wise addition
    act_xg = Activation('relu')(add_xg)
    
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)  # 1x1 convolution to compute attention coefficients
    sigmoid_xg = Activation('sigmoid')(psi)          # Sigmoid to get attention weights
    return multiply([sigmoid_xg, x])  # Multiply attention coefficients with the input feature map

def attention_unet(input_shape=(128, 128, 3)):
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    
    # Decoder
    g = UpSampling2D((2, 2))(c3)
    attn1 = attention_gate(c2, g, 128)  # Attention gate applied on c2
    u1 = concatenate([g, attn1])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    g = UpSampling2D((2, 2))(c4)
    attn2 = attention_gate(c1, g, 64)  # Attention gate applied on c1
    u2 = concatenate([g, attn2])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
