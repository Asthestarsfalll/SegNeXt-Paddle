# SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation

The Paddle Implementation of [SegNeXt](https://arxiv.org/pdf/2209.08575.pdf) based on [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg).

## Start

In order to make you focus on the key points, redundant files have been removed.

LPSNet is defined in `paddleseg/models/lpsnet.py`


### Installation

```bash
pip install -r requirements.txt
pip install -v -e . 
```

### Train

Training with a single GPU.

```bash
python tools/train.py --config configs/segnext/segnext_mscan_t_cityscapes_1024x1024_160k.yml --do_eval --use_vdl --log_iter 100 --save_interval 8000 --save_dir output
```

Training with 4 GPUs.

```bash
python -m paddle.distributed.launch tools/train.py --config configs/segnext/segnext_mscan_t_cityscapes_1024x1024_160k.yml  --do_eval --use_vdl --log_iter 100 --save_interval 8000 --save_dir output
```

### Evaluate

```bash
python tools/val.py --config configs/segnext/segnext_mscan_t_cityscapes_1024x1024_160k.yml --model_path {your_model_path}
```

### Test TIPC

Prepare dataset(very small)

```bash
bash test_tipc/prepare.sh ./test_tipc/configs/segnext/train_infer_python.txt 'lite_train_lite_infer'
```

Test TIPC.

```bash
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/segnext/train_infer_python.txt 'lite_train_lite_infer'
 ```

## Performance

### Cityscapes

|  Model  | Backbone | Resolution | Training Iters |  mIoU  | mIoU (flip) | mIoU (ms+flip) |             Links               |
| :-----: | :------: | :--------: | :------------: | :----: | :---------: | :------------: | :-----------------------------: |
| SegNeXt | MSCAN_T  | 1024x1024  |     160000     | 81.04% |   81.20%    |     81.43%     | [model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/segnext_mscan_t_cityscapes_1024x1024_160k/model.pdparams) \|[log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/segnext_mscan_t_cityscapes_1024x1024_160k/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=5df774c3adc7bc105bc29cd400ccf02b)  |
| SegNeXt | MSCAN_S  | 1024x1024  |     160000     | 81.33% |   81.44%    |     81.47%     | [model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/segnext_mscan_s_cityscapes_1024x1024_160k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/segnext_mscan_s_cityscapes_1024x1024_160k/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/index?id=5d9b1c1a72007c17b380de03bb292f2e) |
| SegNeXt | MSCAN_B  | 1024x1024  |     160000     | 82.74% |   82.84%    |     83.01%     | [model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/segnext_mscan_b_cityscapes_1024x1024_160k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/segnext_mscan_b_cityscapes_1024x1024_160k/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=8185cb34b3f78d12e7e1c51aba13dbe7) |
| SegNeXt | MSCAN_L  | 1024x1024  |     160000     | 83.32% |   83.38%    |     83.60%     | [model](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/segnext_mscan_l_cityscapes_1024x1024_160k/model.pdparams) \| [log](https://paddleseg.bj.bcebos.com/dygraph/cityscapes/segnext_mscan_l_cityscapes_1024x1024_160k/train.log) \| [vdl](https://www.paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=ce122892e0e341a3ad4910c704cb11b8)|

**Note**: In the current implementation, we found some potential issues that could cause training of SegNeXt with backbone MSCAN_T (denoted as SegNeXt-MSCAN_T) to crash when using the multi-card training setup from the original paper. As a work around, we did not use different settings of learning rate and weight decay for different layers. At the same time, we amplified the global learning rate by 10 times. With this setup, we obtained the above results of SegNeXt-MSCAN_T. 

## Citation

```
@article{guo2022segnext,
  title={SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation},
  author={Guo, Meng-Hao and Lu, Cheng-Ze and Hou, Qibin and Liu, Zhengning and Cheng, Ming-Ming and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2209.08575},
  year={2022}
}


@article{guo2022visual,
  title={Visual Attention Network},
  author={Guo, Meng-Hao and Lu, Cheng-Ze and Liu, Zheng-Ning and Cheng, Ming-Ming and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2202.09741},
  year={2022}
}


@inproceedings{
    ham,
    title={Is Attention Better Than Matrix Decomposition?},
    author={Zhengyang Geng and Meng-Hao Guo and Hongxu Chen and Xia Li and Ke Wei and Zhouchen Lin},
    booktitle={International Conference on Learning Representations},
    year={2021},
}


@misc{liu2021paddleseg,
      title={PaddleSeg: A High-Efficient Development Toolkit for Image Segmentation},
      author={Yi Liu and Lutao Chu and Guowei Chen and Zewu Wu and Zeyu Chen and Baohua Lai and Yuying Hao},
      year={2021},
      eprint={2101.06175},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{paddleseg2019,
    title={PaddleSeg, End-to-end image segmentation kit based on PaddlePaddle},
    author={PaddlePaddle Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleSeg}},
    year={2019}
}
```
