# Tie-Aware Hashing
This repository contains Matlab implementation for the following paper:

"Hashing as Tie-Aware Learning to Rank",  
    Kun He, Fatih Cakir, Sarah Adel Bargal, and Stan Sclaroff.
    IEEE CVPR, 2018 ([arXiv](https://arxiv.org/abs/1705.08562))

If you use this code in your research, please cite:
```
@inproceedings{He_2018_TALR,
  title={Hashing as Tie-Aware Learning to Rank},
  author={Kun He and Fatih Cakir and Sarah Adel Bargal and Stan Sclaroff},
  booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month={June}, 
  year={2018}
}
```

## Preparation
- Create or symlink a directory `cachedir` under the root directory to hold experimental results
- Install or symlink [VLFeat](http://www.vlfeat.org/)  at `./vlfeat` (for computing performance metrics)
- Install or symlink [MatConvNet](http://www.vlfeat.org/matconvnet/) at `./matconvnet` (for training CNNs)
- In the `data` directory, run `download_models.sh` to download ImageNet-pretrained CNNs (VGG-F and AlexNet)
- For CIFAR-10: in `data`, run `download_cifar.sh`. This will download the precomputed IMDB format used in our experiments.
- For NUS-WIDE: in `data`, run `download_nuswide.sh` to download the original images.
Decompress the downloaded .zip file into `data/NUSWIDE_images`.
- For ImageNet100: we use the images and train/test split provided by [HashNet](http://github.com/thuml/HashNet).
Please download the `imagenet.tar.gz` file [here](https://drive.google.com/drive/folders/0B7IzDz-4yH_HOXdoaDU4dk40RFE) 
and decompress it into `data/ImageNet`.
Then, get the split files [here](https://github.com/thuml/HashNet/tree/master/caffe/data/imagenet).
The resulting directory should have the following structure:
```
data/ImageNet/
	|- image
	|- val_image
	|- train.txt
	|- test0.txt ... text7.txt
	|- database0.txt ... database7.txt
```

## Usage
- In the root folder, run `startup.m`
- To (approximately) replicate results in the paper, run one of the `run_*.m` files.
For example, `run_cifar_s1(32)` will run the Setting 1 experiment on the CIFAR-10 dataset, with 32-bit hash codes, using the default parameters therein.
- Alternatively, directly run the files in `+demo/` with your parameter choices.
See `main/get_opts.m` for the parameters.

## License
MIT License, see `LICENSE`

## Contact
For questions/comments, feel free to contact:

hekun@bu.edu

## Notes
- Currently only AP experiments are included. NDCG experiments will be added soon.
- We provide the implementation of a simplified version of the tie-aware AP 
(in `apr_s_forward.m` and `apr_s_backward.m`), which attains similar performance 
compared to the original version, but is much simpler to implement.
The derivations will be added soon.
- This implementation is partly based on [MIHash](http://github.com/fcakir/mihash).
