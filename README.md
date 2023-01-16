# SupReMap-Imaginaire

This repository is based on https://github.com/NVlabs/imaginaire, and has been extended to synthesize high-resolution satellite images based on data generated with the SupReMap project, as described in https://github.com/gsaltintas/RemoteSensingData.

## Setup

Please follow the setup instructions provided below by the original Imaginaire authors.

## Dataset

Download our datasets into `/data/` before training:

```
cd dataset
wget https://algvrithm.com/files/supremap/supremap_lawin_swisstopo_dataset_real.zip
unzip supremap_lawin_swisstopo_dataset_real.zip
```

## Training

This repository has been tested using a CUDA GPU with 24GB of VRAM available.

### Train on Swisstopo data with style encoder (9 feature channels):

`python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 train.py --config=configs/projects/pix2pixhd/supremap/supremap_swisstopo_256_with_style_enc_9_feat_ch.yaml`

### Train on Swisstopo data without style encoder:

`python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 train.py --config=configs/projects/pix2pixhd/supremap/supremap_swisstopo_256_without_style_enc.yaml`

## Inference

### Inference on Swisstopo data with style encoder (9 feature channels):

`python inference.py --single_gpu --config=configs/projects/pix2pixhd/supremap/supremap_swisstopo_256_with_style_enc_9_feat_ch.yaml --output_dir=projects/pix2pixhd/output/supremap_imaginaire_swisstopo_with_style_encoder --checkpoint=<path_to_your_checkpoint>`

Generated images will be saved into `projects/pix2pixhd/output/supremap_imaginaire_swisstopo_with_style_encoder`.

### Inference on Swisstopo data without style encoder:

`python inference.py --single_gpu --config=configs/projects/pix2pixhd/supremap/supremap_swisstopo_256_without_style_enc.yaml --output_dir=projects/pix2pixhd/output/supremap_imaginaire_swisstopo_without_style_encoder --checkpoint=<path_to_your_checkpoint>`

Generated images will be saved into `projects/pix2pixhd/output/supremap_imaginaire_swisstopo_without_style_encoder`.



## Pretrained Models

This repository will automatically download and use a Cityscapes-1K-pretrained checkpoint when training is started, courtesy to the original Imaginaire authors.


### Model trained on Swisstopo data with style encoder (9 feature channels) for 16.5K iterations:

`https://algvrithm.com/files/supremap/pix2pixhd_with_style_encoder_iter_16500.pt`

### Model trained on Swisstopo data without style encoder for 18.5K iterations:

`https://algvrithm.com/files/supremap/pix2pixhd_without_style_encoder_iter_18500.pt`

## Known Issues

- "Discriminator overflowed"/"Generator overflowed" may get printed throughout training: https://github.com/NVlabs/imaginaire/issues/126
- "Broken pipe" traceback messages may occasionally get printed throughout training.
- On rare occasions, the model diverges during training. Monitor the visualizations and restart training from the last stable checkpoint if necessary. 
- `cusolverDn.h` not found during setup: search for and install matching version from http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/, then run `export CPLUS_INCLUDE_PATH=<your_cuda_path>/targets/x86_64-linux/include/:$CPLUS_INCLUDE_PATH`


## Results

We provide the following Peak Signal-Noise Ratio (PSNR), Structural Similarity Index (SSIM) and Frechet Inception Distance (FID) results achieved on the SupReMap Swisstopo dataset for reference, calculated using `scripts/calculate_metrics.py`. Visualizations are available at https://algvrithm.com/supremap-vis-v1/.

### With style encoder (9 feature channels):

``` 
    PSNRs (⬆️):           SSIMs (⬆️):            FIDs (⬇️):   

count  557.000000      count  557.000000      count  557.000000
mean    13.628011      mean     0.188923      mean     0.172918
std      1.226479      std      0.075893      std      0.037491
min      9.797507      min      0.069866      min      0.053354
25%     12.887375      25%      0.143539      25%      0.146219
50%     13.567724      50%      0.168461      50%      0.170689
75%     14.222213      75%      0.205571      75%      0.195337
max     19.902980      max      0.621150      max      0.300574
```

### Without style encoder:

```  
    PSNRs (⬆️):           SSIMs (⬆️):            FIDs (⬇️):   
                                                                                              
count  557.000000      count  557.000000      count  557.000000
mean    13.257297      mean     0.185602      mean     0.158049
std      1.191875      std      0.077652      std      0.039535
min      9.903153      min      0.076972      min      0.065973
25%     12.489568      25%      0.139178      25%      0.130352
50%     13.200566      50%      0.165121      50%      0.155623
75%     13.981369      75%      0.200457      75%      0.181643
max     17.848160      max      0.625442      max      0.334908
```


## Pretrained Models

### pix2pixHD with style encoder (9 feature channels):

`https://algvrithm.com/files/supremap/pix2pixhd_with_style_encoder_iter_16500.pt`

Use with configuration file

`configs/projects/pix2pixhd/supremap/supremap_swisstopo_256_with_style_enc_9_feat_ch.yaml`

### pix2pixHD without style encoder:

`https://algvrithm.com/files/supremap/pix2pixhd_without_style_encoder_iter_18500.pt`

Use with configuration file

`configs/projects/pix2pixhd/supremap/supremap_swisstopo_256_without_style_enc.yaml`


# Original Imaginaire README

<img src="imaginaire_logo.svg" alt="imaginaire_logo.svg" height="360"/>

# Imaginaire
### [Docs](http://deepimagination.cc/) | [License](LICENSE.md) | [Installation](INSTALL.md) | [Model Zoo](MODELZOO.md)

Imaginaire is a [pytorch](https://pytorch.org/) library that contains
optimized implementation of several image and video synthesis methods developed at [NVIDIA](https://www.nvidia.com/en-us/).

## License

Imaginaire is released under [NVIDIA Software license](LICENSE.md).
For commercial use, please consult [NVIDIA Research Inquiries](https://www.nvidia.com/en-us/research/inquiries/).

## What's inside?

[![IMAGE ALT TEXT](http://img.youtube.com/vi/jgTX5OnAsYQ/0.jpg)](http://www.youtube.com/watch?v=jgTX5OnAsYQ "Imaginaire")

We have a tutorial for each model. Click on the model name, and your browser should take you to the tutorial page for the project.

### Supervised Image-to-Image Translation

|Algorithm Name                               | Feature                                                                                                         | Publication                                                   |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------:|
|[pix2pixHD](projects/pix2pixhd/README.md)     | Learn a mapping that converts a semantic image to a high-resolution photorealistic image.                       |    [Wang et. al. CVPR 2018](https://arxiv.org/abs/1711.11585) |
|[SPADE](projects/spade/README.md)             | Improve pix2pixHD on handling diverse input labels and delivering better output quality.                        |    [Park et. al. CVPR 2019](https://arxiv.org/abs/1903.07291) |


### Unsupervised Image-to-Image Translation


|Algorithm Name                               | Feature                                                                                                         | Publication                                                   |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------:|
|[UNIT](projects/unit/README.md)               | Learn a one-to-one mapping between two visual domains.                                                          |    [Liu et. al. NeurIPS 2017](https://arxiv.org/abs/1703.00848) |
|[MUNIT](projects/munit/README.md)             | Learn a many-to-many mapping between two visual domains.                                                        |    [Huang et. al. ECCV 2018](https://arxiv.org/abs/1804.04732) |
|[FUNIT](projects/funit/README.md)             | Learn a style-guided image translation model that can generate translations in unseen domains.                  |    [Liu et. al. ICCV 2019](https://arxiv.org/abs/1905.01723) |
|[COCO-FUNIT](projects/coco_funit/README.md)   | Improve FUNIT with a content-conditioned style encoding scheme for style code computation.                      |    [Saito et. al. ECCV 2020](https://arxiv.org/abs/2007.07431) |


### Video-to-video Translation


|Algorithm Name                               | Feature                                                                                                         | Publication                                                   |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------:|
|[vid2vid](projects/vid2vid/README.md)         | Learn a mapping that converts a semantic video to a photorealistic video.                                       |    [Wang et. al. NeurIPS 2018](https://arxiv.org/abs/1808.06601) |
|[fs-vid2vid](projects/fs_vid2vid/README.md)   | Learn a subject-agnostic mapping that converts a semantic video and an example image to a photoreslitic video.  |    [Wang et. al. NeurIPS 2019](https://arxiv.org/abs/1808.06601) |


### World-to-world Translation


|Algorithm Name                               | Feature                                                                                                         | Publication                                                   |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------:|
|[wc-vid2vid](projects/wc_vid2vid/README.md)   | Improve vid2vid on view consistency and long-term consistency.                                                  |    [Mallya et. al. ECCV 2020](https://arxiv.org/abs/2007.08509) |
|[GANcraft](projects/gancraft/README.md)   | Convert semantic block worlds to realistic-looking worlds.                                                  |    [Hao et. al. ICCV 2021](https://arxiv.org/abs/2104.07659) |



