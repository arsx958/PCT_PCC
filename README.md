# **Transformer and Upsampling-Based Point Cloud Compression**

Learning-based point cloud compression has exhibited superior coding performance over the traditional methods such as MEPG G-PCC. Considering that conventional point cloud representation formats (e.g., octree or voxel) will introduce additional errors and affect the reconstruction quality, we directly use the point-based representation and develop a framework that leverages transformer and upsampling techniques for point cloud compression. To extract latent features that well characterize an input point cloud, we build an end-to-end learning framework: at the encoder side, we leverage cascading transformers to extract and enhance useful features for entropy coding; At the decoder side, in addition to the transformers, an upsampling module utilizing both coordinates and features is devised to reconstruct the point cloud progressively. Experimental results demonstrate that the proposed method achieves the best coding performance against state-of-the-art point-based methods, e.g., > 1 dB D1 and D2 PSNR at bitrate 0.10 bpp and more visually pleasing reconstructions. Extensive ablation studies also confirm the effectiveness of transformer and upsampling modules.

## News

- 2022.10.14 published on APCCPA '22: Proceedings of the 1st International Workshop on Advances in Point Cloud Compression, Processing and Analysis.

## Requirments

- python 3.7 or 3.8


- cuda 10.2 or 11.0


- pytorch 1.7 or 1.8

- Pretrained Models: [Pretrained Models](https://drive.google.com/file/d/13M4T3WT1b9OwUQA7bd8vdbPomYQNIebu/view?usp=sharing)

- Training dataset and Testdata: [ShapeNetCore.v2](https://github.com/AnTao97/PointCloudDatasets)

## Usage

### Testing

```shell
sudo chmod 777 utils/pc_error
python test.py --dataset_path='dataset_path'
```

You can unzip pretrained models to `log`.

### Training

```shell
 python train.py --dataset_path='dataset_path'
```

### If you find our project useful, please cite:

```shell
@inproceedings{zhang2022transformer,
  title={Transformer and Upsampling-Based Point Cloud Compression},
  author={Zhang, Junteng and Liu, Gexin and Ding, Dandan and Ma, Zhan},
  booktitle={Proceedings of the 1st International Workshop on Advances in Point Cloud Compression, Processing and Analysis},
  pages={33--39},
  year={2022}
}
```

Our sincere gratitude is directed to the authors of paper **"Point cloud geometry compression via neural graph sampling"** (https://github.com/linyaog/point_based_pcgc) for discussing experiments in detail and providing open source code.

```shell
@inproceedings{gao2021point,
  title={Point cloud geometry compression via neural graph sampling},
  author={Gao, Linyao and Fan, Tingyu and Wan, Jianqiang and Xu, Yiling and Sun, Jun and Ma, Zhan},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={3373--3377},
  year={2021},
  organization={IEEE}
}
```

