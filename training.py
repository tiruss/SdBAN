import numpy as np
from skimage.transform import resize
from skimage.io import imread, imsave
import random
from utils import random_crop
from network import BASNet


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


img = sorted(glob.glob("saliency_dataset/duts_dataset/DUTS-TR/DUTS-TR-Image/*.jpg"))
mask = sorted(glob.glob("saliency_dataset/duts_dataset/DUTS-TR/DUTS-TR-Mask/*.png"))

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


model = BASNet()



