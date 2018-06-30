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
- Install or symlink [MatConvNet](http://www.vlfeat.org/matconvnet/) at `./matconvnet` (for training CNNs)
- Install or symlink [VLFeat](http://www.vlfeat.org/)  at `./vlfeat`. 
**Note**: this is only necessary for computing the regular *tie-agnostic* AP metric.
We provide efficient implementation for the *tie-aware* metrics in `+eval`.
- For ImageNet-pretrained models: in the `data` directory, run `download_models.sh` to download pretrained CNNs (VGG-F and AlexNet).
- For CIFAR-10: in `data`, run `download_cifar.sh`. This will download the precomputed IMDB format used in our experiments.
- For NUS-WIDE: see `data/NUSWIDE/README.md`.
- For ImageNet100: see `data/ImageNet/README.md`.

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
