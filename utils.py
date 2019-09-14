import random


def random_crop(img, mask, width, height):

    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)

    new_img = img[y:y + height, x:x + width, :]
    new_mask = mask[y:y + height, x:x + width, :]

    ratio = random.random()

    # Random Flip
    if ratio > 0.5:
        new_img = new_img[:, ::-1, :]
        new_mask = new_mask[:, ::-1, :]

    return new_img, new_mask
