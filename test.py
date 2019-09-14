import argparse
from tensorflow.keras import Model
from network import BASNet
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import multi_gpu_model

from skimage.transform import resize
from skimage.io import imread

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", required=True)
    parser.add_argument("--input_img", help="For single image prediction", required=False)
    parser.add_argument("--input_dir", help="For multiple image prediction", required=False)
    parser.add_argument("--output_folder", default=os.makedirs("results", exist_ok=True))

    args = parser.parse_args()

    img = np.expand_dims(resize(plt.imread(args.input_img), (224, 224)), 0)

    model = BASNet().build_model()
    print(model.summary())

    # model = multi_gpu_model(BASNet().build_model(), gpus=2)
    model.load_weights(args.weight)
    print("Load Model successfully..")

    prediction = np.squeeze(model.predict(img), 0)

    plt.imsave(args.output_folder +"/"+ args.input_img, prediction[:, :, 0], cmap="gray")


