## ImageNet100 Download Instructions

We use the images and train/test split provided by [HashNet](http://github.com/thuml/HashNet).
Please download the `imagenet.tar.gz` file [at this link](https://drive.google.com/drive/folders/0B7IzDz-4yH_HOXdoaDU4dk40RFE) 
and decompress it in this directory.
Then, get the split files [at this link](https://github.com/thuml/HashNet/tree/master/caffe/data/imagenet).

This directory should have the following structure:
```
data/ImageNet/
	|- image/
	|- val_image/
	|- train.txt
	|- test0.txt ... text7.txt
	|- database0.txt ... database7.txt
```
