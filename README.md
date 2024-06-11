# Robust Map Fusion with Visual Attention Utilizing Multi-agent Rendezvous [ICRA 2023]
[[Paper Link]](https://ieeexplore.ieee.org/abstract/document/10161072)
## About
This repository contains the following contents:
* the implementation of Frame-Error-Estimator from the paper, which aligns the origins of two occupancy grids.
  
* the custom dataset and model weight files to train models and reproduce the results.
  
* the implementation of the baseline method from [*Fast and accurate map merging for multi-robot systmes*](https://link.springer.com/article/10.1007/s10514-008-9097-4).

The implementation of Vision Transformer (`vit_pytorch/`) is cloned from [lucidrains](https://github.com/lucidrains/vit-pytorch) and modified to only return encoded features.

## Installation & Dependencies
* **Model dependencies**
```bash
foo@bar:~$ pip install torch torchvision
foo@bar:~$ pip install einops opencv-python scikit-learn
```

* **Interface dependencies**
```bash
foo@bar:~$ pip install tqdm matplotlib
```

You can reference whole dependencies in [requirement.txt](./requirements.txt), a freezed library list from the lastly tested environment.

## How to use
```bash
foo@bar:~$ git clone https://github.com/qpwodlsqp/Map-Merger.git

foo@bar:~$ cd Map-Merger

foo@bar:~/Map-Merger$ unzip MyMap.zip                                                     # download dataset from the link below

foo@bar:~/Map-Merger$ python train.py --model_type vit --use_lie_regress --use_rec_loss   # train model variants

foo@bar:~/Map-Merger$ python test.py --map_viz --plot_viz                                 # test baseline method and variants
```
Set `model_type`, `use_lie_regress`, and `use_rec_loss` to configure model variants.

Set `map_viz` and `plot_viz` to visualize aligned maps and figure result in the paper.

## Dataset & Weight
* [[Google Drive Link]](https://drive.google.com/drive/folders/12eSXxTzi4RXTpjUzEyktL441LHIsjnEI?usp=drive_link)

Download every file and unzip `MyMap.zip` in a working directory. Weight files are provided to reproduce the reported result in the paper.
```
MyMap.zip   # 30391 pairs in the training set, 357 pairs of images in the test set
├── train
│   ├── 0_cam.png
│   ├── 0_tar.png
│   ├── ...
├── test
│   ├── 0_cam.png
│   ├── 0_tar.png
│   ├── ...
weight
├── mergenet_cnn_rec-x_lie-x.pth
├── mergenet_cnn_rec-x_lie-o.pth
├── mergenet_cnn_rec-o_lie-x.pth
├── mergenet_cnn_rec-o_lie-o.pth
├── mergenet_vit_rec-x_lie-x.pth
├── mergenet_vit_rec-x_lie-o.pth
├── mergenet_vit_rec-o_lie-x.pth
└── mergenet_vit_rec-o_lie-o.pth
```

## Citation
If you find this repository is useful for your project, please consider to cite it as belows.
```bibtex
@inproceedings{kim2023robust,
  title={Robust Map Fusion with Visual Attention Utilizing Multi-agent Rendezvous},
  author={Kim, Jaein and Han, Dong-Sig and Zhang, Byoung-Tak},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={2062--2068},
  year={2023},
  organization={IEEE}
}
```
