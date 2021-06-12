from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model



###################### U-Net model ##########################

class UNet_model():

    def __init__(self):
        # Image shape
        self.IMAGE_WIDTH = 128
        self.IMAGE_HEIGHT = 128
        self.IMAGE_CHANNELS = 3

        self.image_shape = (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS)
        self.num_filters = 32

        # Number of classes
        self.num_classes = 13

    """U-Net Model"""

    def build_unet(self):
        # Convolutional downsampling block
        def encoder_block(input_layer, num_filters=32, dropout=0, max_pooling=True):
            # Add downsampling layer
            encoder = Conv2D(num_filters, (3 ,3),
                             padding="same",
                             activation="relu",
                             kernel_initializer='he_normal')(input_layer)
            encoder = Conv2D(num_filters, (3 ,3),
                             padding="same",
                             activation="relu",
                             kernel_initializer='he_normal')(encoder)
            # conditionally add dropout
            if dropout > 0:
                encoder = Dropout(rate=dropout)(encoder)

            if max_pooling:
                next_encoder = MaxPooling2D(pool_size=(2 ,2))(encoder)

            else:
                next_encoder = encoder

            skip_coonection = encoder

            return next_encoder, skip_coonection

        # Convolutional upsampling block
        def decoder_block(input_layer, skip_input, num_filters=32, dropout=True):
            # Add upsampling layer
            upsample_layer = Conv2DTranspose(num_filters, (3 ,3),
                                             strides=(2 ,2),
                                             padding="same")(input_layer)
            # Merge with skip connection
            merge = concatenate([upsample_layer, skip_input])
            decoder = Conv2D(num_filters, (3 ,3),
                             padding="same",
                             activation="relu",
                             kernel_initializer='he_normal')(merge)
            decoder = Conv2D(num_filters, (3 ,3),
                             padding="same",
                             activation="relu",
                             kernel_initializer='he_normal')(decoder)

            return decoder


        # define the generator model
        # Image input
        img_in = Input(shape = self.image_shape)

        # encoder model (downsampling)
        d1 = encoder_block(img_in, self.num_filters)
        d2 = encoder_block(d1[0], self.num_filters *2)
        d3 = encoder_block(d2[0], self.num_filters *4)
        d4 = encoder_block(d3[0], self.num_filters *8, dropout=0.3)
        d5 = encoder_block(d4[0], self.num_filters *16, dropout=0.3, max_pooling=False)

        # decoder model (Upsampling)
        # add skip connection
        u1 = decoder_block(d5[0], d4[1], self.num_filters *8)
        u2 = decoder_block(u1, d3[1], self.num_filters *4)
        u3 = decoder_block(u2, d2[1], self.num_filters *2)
        u4 = decoder_block(u3, d1[1], self.num_filters)

        conv_lr10 = Conv2D(self.num_filters, (3 ,3),
                           padding = "same",
                           activation="relu",
                           kernel_initializer = "he_normal")(u4)
        # Output layer
        # Add kernel size of 1
        img_out = Conv2D(self.num_classes, (1 ,1), padding="same")(conv_lr10)

        # define model
        model = Model(inputs=img_in, outputs=img_out)
        model.summary()

        return model