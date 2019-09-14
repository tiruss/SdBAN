from network import SdBAN
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_folder", type=str, default="./datasets/DUTS-TR/DUTS-TR-Image/")
    parser.add_argument("--label_folder", type=str, default="./datasets/DUTS-TR/DUTS-TR-Mask/")
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_gpu", default=2, type=int)

    args = parser.parse_args()

    SdBAN = SdBAN()
    SdBAN.train(epoch=args.epoch, batch_size=args.batch_size, gpu=args.num_gpu,
                 img_dir=args.img_folder, label_dir=args.label_folder)
