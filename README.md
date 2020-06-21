# SdBAN: Salient Object Detection Using Bilateral Attention Network with Dice Coefficient Loss (IEEE ACCESS) [[Link]](https://ieeexplore.ieee.org/document/9107080)

Tensorflow based keras implementation of "SdBAN: Salient Object Detection Using Bilateral Attention Network with Dice Coefficient Loss"

### Qualitative Evaluation

<img src="readme_image/qualitative.PNG">

### Quantative Eavaluation

<img src="readme_image/quantative.PNG">

<img src="readme_image/PR_F_curve.png">

## Getting Started
### Installation

- Clone thos repository
```
git clone https://github.com/tiruss/Salient_Code.git
```

- You can install all the dependencies by
```
pip install -r requirements.txt
```

### Download datasets

- Download training datasets [[DUTS-TR]](http://saliencydetection.net/duts/download/DUTS-TR.zip) from the link 

- Download [[HKU-IS]](https://sites.google.com/site/ligb86/hkuis) for test from the link 

- Other datasets can download from the link [[sal_eval_toolbox]](https://github.com/ArcherFMY/sal_eval_toolbox) Thank you for the awesome evaluation toolbox!

### Run experiments from pretrained weight

- Download pretrained weight from the link 

- [[Google drive]](https://drive.google.com/drive/folders/1uaMF84-0zohQ2rHi9mF3xWX4D4FgmN0Y?usp=sharing) [[Baidu drive]]() Baidu drive will be updated soon.

- Run test.py
```
python test.py --weight [pretrained weight] --input_dir [test_img_dir] --output_folder "outputs"
```
- Pre-computed salinecy maps can download from the link

- [[Google drive]](https://drive.google.com/open?id=15aWO3ig2XJajUxvjdx8yiGgeo8MNi-Nj) [[Baidu drive]]() Baidu drive will be updated soon.

### Train from scratch

- DUTS-TR is our traning set for pair comparison

- Run train.py

```
python train.py --img_folder [DUTS-TR img dir] --label_folder [DUTS-TR label dir] --epoch --batch_size --num_gpu
```

### Citation

```
@ARTICLE{9107080,
  author={D. {Kang} and S. {Park} and J. {Paik}},
  journal={IEEE Access}, 
  title={SdBAN: Salient Object Detection Using Bilateral Attention Network With Dice Coefficient Loss}, 
  year={2020},
  volume={8},
  number={},
  pages={104357-104370},}
```
