# CLIPSORT: Multi-Object Tracking with CLIP Features

A from-scratch DeepSORT tracker. Ultralytics YOLO (or ByteTrack YOLOX-X) proposes boxes; CLIP, DINOv2, or ResNet embeddings keep identities stable.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

<p align="center">
    <img src="demo.gif" alt="CLIPSORT Demo" width="800"/>
</p>

- Demo video: https://youtu.be/DlhetgxPCvI
- Extra models / assets: https://drive.google.com/drive/folders/1C9SY023XiH443Ka4spOZOFqxh2otMKbx?usp=sharing

## Features

- **From-scratch DeepSORT** — Kalman filter, cascade matching, Hungarian assignment
- **Appearance backbones** — CLIP ViT-L/14, DINOv2, ResNet50
- **Detectors** — any Ultralytics YOLO / RT-DETR checkpoint, plus ByteTrack YOLOX-X
- **MOT Challenge eval** — MOT16 / MOT17 / MOT20 (`eval_mot.py`)
- **Any COCO class** on Ultralytics weights (see below)

## MOT16 train results

Private detections, CLIP appearance, same DeepSORT hyperparameters. Pedestrian GT only (`class == 1`).

| Detector | Conf | Size | MOTA ↑ | IDF1 ↑ | Recall | Prec | FN | FP | IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11x | 0.40 | 640 | 30.2 | 39.7 | 35.5 | 87.6 | 71,194 | 5,571 | 351 |
| YOLO26x | 0.20 | 640 | 33.4 | 44.3 | 43.7 | 81.8 | 62,154 | 10,757 | 597 |
| YOLO26x | 0.20 | 1280 | 35.0 | 50.0 | 64.0 | 70.0 | 39,706 | 30,833 | 1,184 |
| **YOLOX-X (ByteTrack)** | 0.20 | 800×1440 | **59.9** | **62.0** | **90.8** | 76.3 | **10,169** | 31,175 | 2,984 |

The score is dominated by the detector. COCO YOLO misses small / crowded people (especially MOT16-04). ByteTrack YOLOX-X is a 1-class pedestrian model trained on CrowdHuman + MOT17 train + Cityperson + ETHZ, so **MOT16/17 train MOTA is not a clean hold-out**. Use `--split test` and motchallenge.net for a fair number.

## Tracking classes other than people

**Ultralytics checkpoints** (`yolo26x.pt`, `yolo11x.pt`, `rtdetr-x.pt`, …) are COCO 80-class. Set `tracked_entity` to any name in `config.yaml` (`cow`, `car`, `dog`, …). The cows demo uses this path.

**ByteTrack YOLOX-X** (`bytetrack-x`, `bytetrack-x-ablation`, `bytetrack-x-mot20`) is pedestrians only. `tracked_entity` is ignored. Switch back to a `.pt` file for any other class:

```yaml
detection_backbone: "yolo26x.pt"
tracked_entity: "cow"          # or car, person, ...
detection_imgsz: 640
```

## Installation

Python 3.10–3.12. On NVIDIA GPUs, install a CUDA PyTorch build first:

```bash
git clone https://github.com/Mandred009/CLIPSORT-DeepSort_Multiobject_Tracking_Framework_with_CLIP_features.git
cd CLIPSORT-DeepSort_Multiobject_Tracking_Framework_with_CLIP_features

python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Ultralytics weights (`yolo26x.pt`, …) download on first use. `bytetrack-x` downloads the official ~793 MB `.pth.tar` into `weights/` via `gdown` (not committed).

## Usage

```bash
# Video demo (Ultralytics + tracked_entity in config.yaml)
python main.py --video "Test Videos/cows.mp4"

# MOT Challenge (train split has public GT)
python eval_mot.py --data-root /path/to/MOT16 --split train
python eval_mot.py --data-root /path/to/MOT16 --split train --seqs MOT16-05
```

MOT eval writes one MOT Challenge `.txt` per sequence plus `metrics.txt` / `metrics.csv`. The test split has no public GT; submit those files on motchallenge.net.

### ByteTrack YOLOX-X (MOT pedestrians)

```yaml
detection_backbone: "bytetrack-x"
detection_test_size: [800, 1440]
detection_confidence_threshold: 0.20
```

Aliases: `bytetrack-x-ablation`, `bytetrack-x-mot20`.

## Files

```
├── main.py              # Video demo
├── eval_mot.py          # MOT16 / MOT17 / MOT20 evaluation
├── deepsort.py          # Tracker
├── kalman.py            # Kalman filter
├── yolo.py              # Ultralytics YOLO / RT-DETR + detector factory
├── yolox_mot.py         # ByteTrack YOLOX-X pedestrian detector
├── clip_feature.py      # CLIP extractor
├── dino_feature.py      # DINOv2 extractor
├── resnet_feature.py    # ResNet extractor
├── config.yaml          # Detector, class, and association settings
└── requirements.txt
```

## References

```
@inproceedings{wojke2017simple,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  year={2017}
}
@inproceedings{zhang2022bytetrack,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={ECCV},
  year={2022}
}
@inproceedings{ge2021yolox,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  year={2021}
}
@inproceedings{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle={ICML},
  year={2021}
}
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and others},
  journal={arXiv:2304.07193},
  year={2023}
}
```
