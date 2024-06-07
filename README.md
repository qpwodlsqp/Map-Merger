# Robust Map Fusion with Visual Attention Utilizing Multi-agent Rendezvous [ICRA 2023]
[[Paper Link]](https://ieeexplore.ieee.org/abstract/document/10161072)
## About
This repository contains the official implementation of Frame-Error-Estimator, which aligns the coordinates of two occupancy grid maps.
We also attatch the link to download weight files and our dataset to reproduce the reported results or utilize for your own purpose.

The implementation of Vision Transformer is cloned from [lucidrains](https://github.com/lucidrains/vit-pytorch) and modified to return only encoded features.
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

You can reference whole dependencies in requirement.txt, freezed from the library list of last tested environment.

## How to use
```bash
foo@bar:~$ git clone https://github.com/qpwodlsqp/Map-Merger.git
foo@bar:~$ cd Map-Merger
foo@bar:~/Map-Merger$ python train.py --model_type vit --use_lie_regress --use_rec_loss
foo@bar:~/Map-Merger$ python test.py
```

## Dataset & Weight
* [[Google Drive Link]](https://drive.google.com/drive/folders/12eSXxTzi4RXTpjUzEyktL441LHIsjnEI?usp=drive_link)

## Citation
If you find this repository is useful for your project, please consider to cite as belows.
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
