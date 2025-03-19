from keras import layers, Model
import tensorflow as tf

def get_unet_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs, outputs)
    return model

def get_segnet_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(c4)

    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    p5 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(c5)

    # Decoder (using UpSampling2D instead of MaxUnpooling2D)
    u5 = layers.UpSampling2D((2, 2))(p5)
    d5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u5)
    d5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(d5)
    d5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(d5)

    u4 = layers.UpSampling2D((2, 2))(d5)
    d4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u4)
    d4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(d4)
    d4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(d4)

    u3 = layers.UpSampling2D((2, 2))(d4)
    d3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u3)
    d3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(d3)
    d3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(d3)

    u2 = layers.UpSampling2D((2, 2))(d3)
    d2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    d2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(d2)

    u1 = layers.UpSampling2D((2, 2))(d2)
    d1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    d1 = layers.Conv2D(num_classes, (1, 1), activation='softmax', padding='same')(d1)

    return Model(inputs, d1)

def get_deeplab_model(input_shape, num_classes):
    def atrous_spatial_pyramid_pooling(x):
        """Atrous Spatial Pyramid Pooling (ASPP) module."""
        dims = x.shape[1:3]

        # Atrous convolution with different dilation rates
        conv_1x1 = layers.Conv2D(256, (1, 1), padding="same", activation="relu")(x)
        atrous_conv_6 = layers.Conv2D(256, (3, 3), dilation_rate=6, padding="same", activation="relu")(x)
        atrous_conv_12 = layers.Conv2D(256, (3, 3), dilation_rate=12, padding="same", activation="relu")(x)
        atrous_conv_18 = layers.Conv2D(256, (3, 3), dilation_rate=18, padding="same", activation="relu")(x)

        # Global average pooling
        global_avg_pool = layers.GlobalAveragePooling2D()(x)
        global_avg_pool = layers.Reshape((1, 1, global_avg_pool.shape[-1]))(global_avg_pool)
        global_avg_pool = layers.Conv2D(256, (1, 1), padding="same", activation="relu")(global_avg_pool)
        global_avg_pool = layers.UpSampling2D(size=dims, interpolation="bilinear")(global_avg_pool)

        # Concatenate all the features
        x = layers.Concatenate()([conv_1x1, atrous_conv_6, atrous_conv_12, atrous_conv_18, global_avg_pool])
        x = layers.Conv2D(256, (1, 1), padding="same", activation="relu")(x)
        return x

    def encoder(inputs):
        """Encoder network with ASPP."""
        base_model = tf.keras.applications.MobileNetV2(input_tensor=inputs, include_top=False, weights="imagenet")
        skip_connection = base_model.get_layer("block_1_expand_relu").output
        output = base_model.get_layer("block_16_project").output

        # Apply ASPP to the output of the base model
        x = atrous_spatial_pyramid_pooling(output)
        return x, skip_connection

    def decoder(encoder_output, skip_connection):
        """Decoder network with spatial alignment."""
        # Upsample the encoder output to match the spatial dimensions of the skip connection
        x = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(encoder_output)

        # Reduce the number of channels in the skip connection for better blending
        skip_connection = layers.Conv2D(48, (1, 1), padding="same", activation="relu")(skip_connection)

        # Ensure the spatial dimensions of both tensors match
        skip_connection = layers.Resizing(x.shape[1], x.shape[2])(skip_connection)

        # Concatenate the upsampled encoder output with the skip connection
        x = layers.Concatenate()([x, skip_connection])

        # Add convolutional layers to refine the combined features
        x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
        x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)

        # Final upsampling to match the input image size
        x = layers.UpSampling2D(size=(input_shape[0] // x.shape[1], input_shape[1] // x.shape[2]), interpolation="bilinear")(x)

        # Output layer with softmax activation for semantic segmentation
        x = layers.Conv2D(num_classes, (1, 1), padding="same", activation="softmax")(x)
        return x

    inputs = layers.Input(shape=input_shape)
    encoder_output, skip_connection = encoder(inputs)
    outputs = decoder(encoder_output, skip_connection)

    model = Model(inputs, outputs)
    return model