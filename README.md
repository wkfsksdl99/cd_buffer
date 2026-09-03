# CD Buffer

This is the official code repository for the CVPR 2026 Highlight paper **"CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection"**.
<img width="1542" height="623" alt="image" src="https://github.com/user-attachments/assets/7819519d-9ac5-4cfc-ac0a-7e607e227563" />
## Installation

The code was tested with the following environment:

- Python 3.7.12
- PyTorch 1.13.1
- torchvision 0.14.1

Create and activate the conda environment

Install the remaining dependencies:

```bash
cd /path/to/cd_buffer
pip install -r requirements.txt
```

The source-trained Faster R-CNN checkpoints can be downloaded from the
repository's [latest GitHub Release](https://github.com/wkfsksdl99/cd_buffer/releases/latest). Place them as
follows:

```bash
mkdir -p models/kitti models/cityscapes
mv /path/to/kitti_best_model.pth models/kitti/best_model.pth
mv /path/to/cityscapes_best_model.pth models/cityscapes/best_model.pth
```

The resulting checkpoint structure is:

```text
cd_buffer/
├── models/
│   ├── kitti/
│   │   └── best_model.pth
│   └── cityscapes/
│       └── best_model.pth
├── cd_buffer_main.py
├── cd_buffer_main_light.py
├── config.py
└── requirements.txt
```


## Dataset

### Dataset sources

Source-domain datasets:

- **KITTI:** download the KITTI 2D object detection data from the [KITTI Object Detection Benchmark](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2).
- **Cityscapes:** download the original clear-weather images from the [Cityscapes Dataset](https://www.cityscapes-dataset.com/).

Target-domain datasets:

- **Rainy/Foggy KITTI:** synthesize the target images following the physics-based weather augmentation pipeline described on the [Inria Weather Augmentation](https://team.inria.fr/rits/computer-vision/weather-augment/) page.
- **Foggy Cityscapes:** prepare the synthetic foggy images following the [Foggy Cityscapes-DBF model adaptation](https://people.ee.ethz.ch/~csakarid/Model_adaptation_SFSU_dense/) setup.
- **ACDC:** download the adverse-condition data and detection annotations from the [ACDC Dataset](https://acdc.vision.ee.ethz.ch/).

Place the datasets under the relative `dataset/` directory or provide their locations at runtime. Use `--dataroot` for KITTI or Cityscapes and `--acdc_dataroot` for the ACDC target dataset.

### KITTI structure

`KITTIDataset` expects the following structure rather than the unmodified KITTI archive layout:

```text
KITTI/
├── train.txt
├── val.txt
├── test.txt
├── initial/
│   └── images/
│       ├── 000000.png
│       └── ...
├── label_2/
│   ├── 000000.txt
│   └── ...
├── rain/
│   └── <severity>/
│       ├── 000000.png
│       └── ...
└── fog/
    └── <severity>/
        ├── 000000.png
        └── ...
```

Each split file contains one image filename, including the `.png` extension, per line:

```text
000000.png
000001.png
```

The clear source image is loaded from `initial/images/<filename>`. A target image is loaded from `<target_task>/<target_severity>/<filename>`, for example `fog/50m/000000.png` or `rain/50mm/000000.png`. Source and target images share the labels in `label_2/`. The label files must follow the standard KITTI object label format; the loader reads the class name and the 2D bounding box fields.

### Cityscapes structure

`CityscapesDataset` expects the clear images, foggy images, and converted object-detection annotations in the following layout:

```text
cityscapes/
├── leftImg8bit_trainvaltest/
│   ├── leftImg8bit/
│   │   ├── train/
│   │   │   └── <city>/
│   │   │       └── <sample>_leftImg8bit.png
│   │   └── val/
│   │       └── <city>/
│   │           └── <sample>_leftImg8bit.png
│   └── annotations/
│       ├── train/
│       │   └── <sample>.json
│       └── val/
│           └── <sample>.json
└── leftImg8bit_foggy/
    ├── train/
    │   └── <city>/
    │       └── <sample>_leftImg8bit_foggy_beta_<severity>.png
    └── val/
        └── <city>/
            └── <sample>_leftImg8bit_foggy_beta_<severity>.png
```

The annotation directory is flat within each split; it does not contain an additional city subdirectory. These JSON files are project-specific object-detection annotations and are not the native Cityscapes label files. Each file must use this format, where `bbox` is `[xmin, ymin, xmax, ymax]`:

```json
{
  "annotations": [
    {
      "class_name": "car",
      "bbox": [100, 120, 300, 260]
    }
  ]
}
```

For `--target_task foggy`, the requested severity must be present in the image filename. For example, `--target_severity 0.01` selects files ending in `_leftImg8bit_foggy_beta_0.01.png`.

### ACDC structure

The ACDC loader uses the official adverse-condition image layout and COCO-style detection JSON files:

```text
ACDC_dataset/
├── rgb_anon/
│   ├── fog/
│   │   ├── train/<sequence>/*.png
│   │   └── val/<sequence>/*.png
│   ├── night/
│   │   ├── train/<sequence>/*.png
│   │   └── val/<sequence>/*.png
│   ├── rain/
│   │   ├── train/<sequence>/*.png
│   │   └── val/<sequence>/*.png
│   └── snow/
│       ├── train/<sequence>/*.png
│       └── val/<sequence>/*.png
└── gt_detection/
    ├── fog/
    │   ├── instancesonly_fog_train_gt_detection.json
    │   └── instancesonly_fog_val_gt_detection.json
    ├── night/
    │   ├── instancesonly_night_train_gt_detection.json
    │   └── instancesonly_night_val_gt_detection.json
    ├── rain/
    │   ├── instancesonly_rain_train_gt_detection.json
    │   └── instancesonly_rain_val_gt_detection.json
    └── snow/
        ├── instancesonly_snow_train_gt_detection.json
        └── instancesonly_snow_val_gt_detection.json
```

The JSON files must contain `images`, `categories`, and `annotations`. Bounding boxes use the COCO `[x, y, width, height]` format. By default, both the ACDC train and validation splits are used for target-domain adaptation. ACDC is supported by the standard runner, not the light runner. Its source domain is Cityscapes, so `--dataroot`, `--weights`, and `--source_stats_path` must point to the Cityscapes source data, checkpoint, and statistics. Use `--acdc_dataroot` for the ACDC target root.

## Run TTA

The standard runner exposes the TTA and BN learning rates and the mask regularization weight. The light runner uses fixed values for these settings.

If the file passed through `--source_stats_path` does not exist, the code extracts source-domain statistics from the source split and saves them automatically. If it exists, the cached statistics are loaded directly.

### KITTI

```bash
python cd_buffer_main.py \
  --dataset kitti \
  --dataroot ./dataset/kitti \
  --source_split train \
  --target_split test \
  --target_task fog \
  --target_severity 50m \
  --weights ./models/kitti/best_model.pth \
  --source_stats_path ./source_stats_kitti.pth \
  --save_dir ./cd_buffer_tta_results/kitti_fog_50m \
  --device YOUR DEVICE
```

For rainy KITTI, change the target task and severity to match the prepared directory:

```bash
python cd_buffer_main.py \
  --dataset kitti \
  --dataroot ./dataset/kitti \
  --source_split train \
  --target_split test \
  --target_task rain \
  --target_severity 50mm \
  --weights ./models/kitti/best_model.pth \
  --source_stats_path ./source_stats_kitti.pth \
  --save_dir ./cd_buffer_tta_results/kitti_rain_50mm \
  --device YOUR DEVICE
```

### Cityscapes

```bash
python cd_buffer_main_light.py \
  --dataset cityscapes \
  --dataroot ./dataset/cityscapes \
  --source_split train \
  --target_split val \
  --target_task foggy \
  --target_severity 0.01 \
  --weights ./models/cityscapes/best_model.pth \
  --source_stats_path ./source_stats_cityscapes.pth \
  --save_dir ./cd_buffer_tta_results_light/cityscapes_foggy_0.01 \
  --device YOUR DEVICE
```

### ACDC

Use the standard runner for ACDC. `--target_task` selects one of `fog`, `night`, `rain`, or `snow`.

```bash
python cd_buffer_main.py \
  --dataset acdc \
  --dataroot ./dataset/cityscapes \
  --acdc_dataroot ./dataset/acdc \
  --source_split train \
  --target_task fog \
  --weights ./models/cityscapes/best_model.pth \
  --source_stats_path ./source_stats_cityscapes.pth \
  --save_dir ./cd_buffer_tta_results/acdc_fog \
  --device YOUR DEVICE
```

During TTA, the progress bar reports the cumulative average mAP50:

```text
TTA: 45%|...| Average mAP50: 0.3321
```

The output directory contains:

```text
<save_dir>/
├── cd_buffer_tta_model.pth
└── cd_buffer_tta_model.txt
```

## References

### Datasets

[1] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. *Vision Meets Robotics: The KITTI Dataset*. The International Journal of Robotics Research, 32(11):1231–1237, 2013.

[2] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. *The Cityscapes Dataset for Semantic Urban Scene Understanding*. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3213–3223, 2016.

[3] Christos Sakaridis, Dengxin Dai, Simon Hecker, and Luc Van Gool. *Model Adaptation with Synthetic and Real Data for Semantic Dense Foggy Scene Understanding*. In Proceedings of the European Conference on Computer Vision, pages 687–704, 2018.

[4] Shirsendu Sukanta Halder, Jean-François Lalonde, and Raoul de Charette. *Physics-Based Rendering for Improving Robustness to Rain*. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019.

[5] Christos Sakaridis, Dengxin Dai, and Luc Van Gool. *ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding*. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10765–10775, 2021.

### Code

[1] Muhammad Jehanzeb Mirza, Pol Jané Soneira, Wei Lin, Mateusz Kozinski, Horst Possegger, and Horst Bischof. *ActMAD: Activation Matching to Align Distributions for Test-Time-Training*. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24152–24161, 2023. ([jmiemirza/ActMAD](https://github.com/jmiemirza/ActMAD))

[2] Jayeon Yoo, Dongkwan Lee, Inseop Chung, Donghyun Kim, and Nojun Kwak. *What, How, and When Should Object Detectors Update in Continually Changing Test Domains?* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23354–23363, 2024. ([natureyoo/ContinualTTA_ObjectDetection](https://github.com/natureyoo/ContinualTTA_ObjectDetection))

[3] Hyeongyu Kim, Geonhui Han, and Dosik Hwang. *Buffer Layers for Test-Time Adaptation*. arXiv preprint arXiv:2510.21271, 2025. ([hyeongyu-kim/Buffer_TTA](https://github.com/hyeongyu-kim/Buffer_TTA))

## Acknowledgment

This code builds upon [ActMAD](https://github.com/jmiemirza/ActMAD), [ContinualTTA_ObjectDetection](https://github.com/natureyoo/ContinualTTA_ObjectDetection), and [Buffer_TTA](https://github.com/hyeongyu-kim/Buffer_TTA). We thank the authors for making their valuable work publicly available.


## License

This project is released under the [Apache License 2.0](LICENSE).

## Citation
If you find our work useful or valuable for your research, please consider citing us:
```text
@InProceedings{Song_2026_CVPR,
    author    = {Song, Youngjun and Kim, Hyeongyu and Hwang, Dosik},
    title     = {CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {15050-15059}
}
```

