import tensorflow as tf
import glob
import numpy as np
import random

from tensorflow.keras import layers, losses, Model, Input, optimizers, initializers, models
from tensorflow.keras.layers import Concatenate, BatchNormalization, Activation, Conv2D, Conv2DTranspose, concatenate, Lambda
from tensorflow.keras.layers import GlobalAveragePooling2D, Multiply, UpSampling2D, Add
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import multi_gpu_model

from skimage.transform import resize, rotate
from skimage.io import imread, imsave

class BASNet():
    def __init__(self):
        # Input shape

        self.output_filter = 32
        self.last_layer = None
        self.skip = None
        self.skip_layer = None

    def build_model(self):

        def DeConv(output_filter, last_layer, skip, skip_layer):
            l = Conv2DTranspose(filters=last_layer.get_shape().as_list()[-1],
                                      kernel_size=(3, 3), strides=(2, 2), padding="same")(last_layer)
            l = BatchNormalization()(l)
            l = Activation("relu")(l)

            if skip:
                skip_layer = concatenate([skip_layer, l], axis=-1)

            l = Conv2D(filters=output_filter, kernel_size=(1, 1), padding="same")(skip_layer)
            l = BatchNormalization()(l)
            l = Activation("relu")(l)

            return l

        def Attention_Module(inputs):
            GAP = Lambda(lambda x: tf.reduce_mean(x, [1, 2], keepdims=True))(inputs)

            l = Conv2D(filters=inputs.get_shape().as_list()[-1], kernel_size=(1, 1))(GAP)
            l = BatchNormalization()(l)
            l = Activation("sigmoid")(l)
            l = Multiply()([inputs, l])

            return l

        def Attention_Fusion_Module(input1, input2, n_filters=128):
            inputs = concatenate([input1, input2], axis=-1)

            a = Conv2D(filters=n_filters, kernel_size=(3, 3), padding="same")(inputs)
            a = BatchNormalization()(a)
            a1 = Activation("relu")(a)

            GAP = Lambda(lambda x: tf.reduce_mean(x, [1, 2], keepdims=True))(a1)
            a = Conv2D(filters=n_filters, kernel_size=(1, 1))(GAP)
            a2 = Activation("relu")(a)
            a = Conv2D(filters=n_filters, kernel_size=(1, 1))(a2)
            a2 = Activation("sigmoid")(a)

            mul = Multiply()([a1, a2])
            add = Add()([a1, mul])

            return add

        # Load Resnet50
        resnet = ResNet50()

        # remove last 2 layers
        resnet._layers.pop()
        resnet._layers.pop()

        input = resnet.input

        resnet_last_layer = resnet.layers[-1].output

        # Encoding..
        block_01 = resnet.get_layer("res4f_branch2c").output  # 14*14
        block_02 = resnet.get_layer("res3d_branch2c").output  # 28*28
        block_03 = resnet.get_layer("res2c_branch2c").output  # 56*56
        block_04 = resnet.get_layer("conv1").output  # 112*112
        block_05 = resnet.get_layer("input_1").output  # 224*224

        """
        Context Attention Path
        """

        Attention_01 = Attention_Module(inputs=block_04)
        Attention_02 = Attention_Module(inputs=block_03)

        global_channel = GlobalAveragePooling2D()(Attention_02)
        Attention_Fusion_01 = Multiply()([Attention_02, global_channel])

        Attention_01 = UpSampling2D(size=(2, 2))(Attention_01)
        Attention_Fusion_01 = UpSampling2D(size=(4, 4))(Attention_Fusion_01)

        Attention_output = concatenate([Attention_01, Attention_Fusion_01], axis=-1)

        """
        Deconding Path
        """
        DeConv_01 = DeConv(self.output_filter*16, last_layer=resnet_last_layer, skip=True, skip_layer=block_01)
        DeConv_02 = DeConv(self.output_filter*8, last_layer=DeConv_01, skip=True, skip_layer=block_02)
        DeConv_03 = DeConv(self.output_filter*4, last_layer=DeConv_02, skip=True, skip_layer=block_03)
        DeConv_04 = DeConv(self.output_filter*2, last_layer=DeConv_03, skip=True, skip_layer=block_04)
        DeConv_05 = DeConv(self.output_filter, last_layer=DeConv_04, skip=True, skip_layer=block_05)

        Final_Fusion = Attention_Fusion_Module(Attention_output, DeConv_05)

        output = Conv2D(1, (1,1), activation="sigmoid")(Final_Fusion)

        return Model(inputs=input, outputs=[output])

    def train(self, epoch=200, batch_size=16, gpu=1):

        def random_crop(img, mask, width, height):
            x = random.randint(0, img.shape[1] - width)
            y = random.randint(0, img.shape[0] - height)

            new_img = img[y:y + height, x:x + width, :]
            new_mask = mask[y:y + height, x:x + width, :]

            ratio = random.random()

            if ratio > 0.5:
                new_img = new_img[:, ::-1, :]
                new_mask = new_mask[:, ::-1, :]

            return new_img, new_mask

        def generator(images, masks, batch_size, input_width=224, input_height=224):

            batch_image = np.zeros((batch_size, input_width, input_height, 3))
            batch_mask = np.zeros((batch_size, input_width, input_height, 1))

            while True:
                for i in range(batch_size):
                    index = int(np.random.choice(len(images), 1))

                    img = imread(images[index])
                    mask = imread(masks[index])
                    mask = np.expand_dims(mask, -1)

                    img = resize(img, (256, 256, 3))
                    mask = resize(mask, (256, 256, 1))

                    width, height = 224, 224

                    batch_image[i], batch_mask[i] = random_crop(img, mask, 224, 224)

                yield batch_image, batch_mask

        img = sorted(glob.glob("datasets/duts_dataset/DUTS-TR/DUTS-TR-Image/*.jpg"))
        mask = sorted(glob.glob("datasets/duts_dataset/DUTS-TR/DUTS-TR-Mask/*.png"))

        # Data Loading..
        train_img, val_img = [], []
        train_mask, val_mask = [], []
        for i, v in enumerate(img):
            index = random.randint(0, len(img) - 1)

            if i < 9000:
                train_img.append(img[index])
                train_mask.append(mask[index])
            else:
                val_img.append(img[index])
                val_mask.append(mask[index])

        # Dice Loss
        def dice_coeff(y_true, y_pred):
            smooth = 1.
            # Flatten
            y_true_f = tf.reshape(y_true, [-1])
            y_pred_f = tf.reshape(y_pred, [-1])
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
            return score

        def dice_loss(y_true, y_pred):
            loss = 1 - dice_coeff(y_true, y_pred)
            return loss

        def bce_dice_loss(y_true, y_pred):
            loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
            return loss

        adam = optimizers.Adam(lr=0.002, decay=0.00005)

        model = self.build_model()

        if gpu > 1:
            model = multi_gpu_model(model, gpus=gpu)
            print("Training using multiple GPUs...")

        model.compile(optimizer=adam, loss=bce_dice_loss, metrics=["mae"])

        file_path = "BASNet_708_.{epoch:02d}.hdf5"

        callback = ModelCheckpoint(filepath=file_path, monitor="val_mean_absolute_error", save_weights_only=True,
                                   save_best_only=True)

        model.fit_generator(generator=generator(train_img, train_mask, batch_size),
                            steps_per_epoch=len(train_img) // batch_size,
                            validation_data=generator(val_img, val_mask, batch_size),
                            validation_steps=len(val_img) // batch_size,
                            epochs=epoch, callbacks=[callback])


if __name__ == "__main__":
    BASNet = BASNet()
    BASNet.train(epoch=200, batch_size=18, gpu=2)











