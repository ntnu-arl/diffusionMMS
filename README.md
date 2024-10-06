# DiffusionMMS
This repository contains the source code for the paper Diffusion-based RGB-D Semantic Segmentation with Deformable Attention Transformer.

## Usage
### Dependencies
The results in the paper is tested on Ubuntu 22.04, pytorch 2.0.1 and CUDA 11.7. To install pytorch and other necessary python library, one can use the following command
```
conda create -n cu117 python=3.10
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```
**Install NATTEN**

To install NATTEN, please follow the instruction in the [NATTEN](https://github.com/SHI-Labs/NATTEN/blob/main/docs/install.md) repository

### Data preparation
Download NYUv2 and SUNRGBD dataset from [Google Drive](https://drive.google.com/drive/u/1/folders/1LeIw-yn7Erk1Zqys2gV5brmgYh7kEsg8) and put them under `data` folder. Each dataset is categorized into train and val split using a text file containing the filename of images file.

To create a list of depth images that have the most proportion of invalid pixels in the datasets, use the following command
```
python -m utils.ranking_data --img_dir <path-to-depth-images-directory> --file <path-to-test-index-file>
```
### Training
Download different variants of UperNet DAT++ backbone from [this repository](https://github.com/LeapLabTHU/DAT-Segmentation) and put them under `pretrained` folder

You now can choose different config file to train different type of models
```
python train.py --config <path-to-config-file>
```
Example 
```
python train.py --config config/nyuv2/standard/ddp_dual_dat_s_mmcv_epoch_100.yaml
```
### Evaluation
Evaluation results of multiple checkpoints can be produced using the following command
```
python eval.py --config <path-to-config-file> --fr <start-epoch> --to <end-epoch>
```
### Visualize
You can visualize the results on each datasets using the following command
```
python test.py --config <path-to-config-file> --epoch <epoch> --show
```

## Citing
If you reference our work in your research, please cite the following paper:

```bibtex
@misc{bui2024diffusionbasedrgbdsemanticsegmentation,
      title={Diffusion-based RGB-D Semantic Segmentation with Deformable Attention Transformer}, 
      author={Minh Bui and Kostas Alexis},
      year={2024},
      eprint={2409.15117},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.15117}, 
}
```
## Contact
For inquiries, feel free to reach out to the authors:
- **Quang Minh Bui**
  
  [Email](mailto:minh.q.bui@ntnu.no) | [GitHub](https://github.com/bqm1111) | [LinkedIn](https://www.linkedin.com/in/quang-minh-bui-250a89131/)

- **Kostas Alexis**

  [Email](mailto:konstantinos.alexis@ntnu.no) | [GitHub](https://github.com/kostas-alexis) | [LinkedIn](https://www.linkedin.com/in/kostas-alexis-67713918/) | [X (formerly Twitter)](https://twitter.com/arlteam)

This research was conducted at the [Autonomous Robots Lab](https://www.autonomousrobotslab.com/), [Norwegian University of Science and Technology (NTNU)](https://www.ntnu.no). 

For more information, visit our website.

## Acknowledgements
Our implementation is partly based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1), [CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) and [DDP](https://github.com/JiYuanFeng/DDP). Thanks for their authors.

This material was supported by the Research Council of Norway under Award NO-338694 and the European Commission under Grant No. 101121321.
