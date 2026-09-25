"""ByteTrack YOLOX-X pedestrian detector (CrowdHuman / MOT mix).

Loads the official 1-class YOLOX-X checkpoints from ByteTrack, not Ultralytics
COCO weights. Inference architecture matches Megvii YOLOX / ByteTrack so the
published .pth.tar files load with strict=True.

Checkpoints (aliases or a local .pth.tar path):
  bytetrack-x           CrowdHuman + MOT17 train + Cityperson + ETHZ, 800x1440
  bytetrack-x-ablation  CrowdHuman + MOT17 half-train, 800x1440
  bytetrack-x-mot20     CrowdHuman + MOT20 train, 896x1600

The MOT17-train mix has seen MOT16/17 train videos. Use it for the MOT test
split, or treat MOT16-train numbers as contaminated.
"""

from __future__ import annotations

import os
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision


# Official ByteTrack Google Drive IDs, plus Hugging Face mirrors.
YOLOX_ZOO = {
    "bytetrack-x": {
        "filename": "bytetrack_x_mot17.pth.tar",
        "test_size": (800, 1440),
        "train_data": "CrowdHuman + MOT17 train + Cityperson + ETHZ",
        "leaks_mot_train": True,
        "gdrive": "1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5",
        "urls": [
            "https://huggingface.co/mrhwl123/YoloxForMOT/resolve/main/bytetrack_x_mot17.pth.tar",
        ],
    },
    "bytetrack-x-ablation": {
        "filename": "bytetrack_x_ablation.pth.tar",
        "test_size": (800, 1440),
        "train_data": "CrowdHuman + MOT17 half-train",
        "leaks_mot_train": True,
        "gdrive": "1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob",
        "urls": [
            "https://huggingface.co/mrhwl123/YoloxForMOT/resolve/main/bytetrack_x_ablation.pth.tar",
        ],
    },
    "bytetrack-x-mot20": {
        "filename": "bytetrack_x_mot20.pth.tar",
        "test_size": (896, 1600),
        "train_data": "CrowdHuman + MOT20 train",
        "leaks_mot_train": True,
        "gdrive": "1HX2_JpMOjOIj1Z9rJjoet9XNy_cCAs5U",
        "urls": [
            "https://huggingface.co/mrhwl123/YoloxForMOT/resolve/main/bytetrack_x_mot20.pth.tar",
        ],
    },
}

_ALIAS_FILES = {meta["filename"]: alias for alias, meta in YOLOX_ZOO.items()}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def is_yolox_backbone(name):
    if not name:
        return False
    lower = str(name).lower()
    basename = os.path.basename(lower)
    return (
        lower in YOLOX_ZOO
        or basename in _ALIAS_FILES
        or lower.endswith(".pth.tar")
        or "yolox" in basename
        or "bytetrack" in basename
    )


def _weights_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


class _DownloadProgress:
    def __init__(self, dest):
        self.dest = dest
        self.last = 0

    def __call__(self, block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size <= 0:
            return
        if downloaded - self.last < 8 * 1024 * 1024 and downloaded < total_size:
            return
        self.last = downloaded
        pct = min(100.0, 100.0 * downloaded / total_size)
        print(
            "  downloading %s  %.1f / %.1f MB (%.0f%%)"
            % (os.path.basename(self.dest), downloaded / 1e6, total_size / 1e6, pct),
            flush=True,
        )


def _download_file(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print("Fetching %s" % url, flush=True)
    urllib.request.urlretrieve(url, tmp, _DownloadProgress(dest))
    os.replace(tmp, dest)


def _download_gdrive(file_id, dest):
    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "Install gdown to download from Google Drive: pip install gdown"
        )
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = "https://drive.google.com/uc?id=%s" % file_id
    print("Fetching Google Drive %s" % file_id, flush=True)
    gdown.download(url, dest, quiet=False)
    if not os.path.isfile(dest) or os.path.getsize(dest) < 1_000_000:
        raise RuntimeError("gdown did not produce a valid file at %s" % dest)


def resolve_yolox_checkpoint(model_name):
    """Return (local_path, zoo_meta_or_None). Downloads aliases on first use."""
    if os.path.isfile(model_name):
        alias = _ALIAS_FILES.get(os.path.basename(model_name))
        return os.path.abspath(model_name), YOLOX_ZOO.get(alias)

    alias = model_name.lower()
    if alias not in YOLOX_ZOO:
        alias = _ALIAS_FILES.get(os.path.basename(model_name))
    if alias not in YOLOX_ZOO:
        raise FileNotFoundError(
            "Unknown YOLOX checkpoint '%s'. Use bytetrack-x, "
            "bytetrack-x-ablation, bytetrack-x-mot20, or a local .pth.tar"
            % model_name
        )

    meta = YOLOX_ZOO[alias]
    dest = os.path.join(_weights_dir(), meta["filename"])
    if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
        return dest, meta

    errors = []
    if meta.get("gdrive"):
        try:
            _download_gdrive(meta["gdrive"], dest)
            return dest, meta
        except Exception as exc:
            errors.append("gdrive:%s (%s)" % (meta["gdrive"], exc))

    for url in meta.get("urls", []):
        try:
            _download_file(url, dest)
            if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
                return dest, meta
        except Exception as exc:
            errors.append("%s (%s)" % (url, exc))
            if os.path.isfile(dest + ".part"):
                os.remove(dest + ".part")

    raise FileNotFoundError(
        "Could not download %s. Tried:\n  %s\n"
        "Install gdown (pip install gdown) or place the official file at %s"
        % (meta["filename"], "\n  ".join(errors) or "(no sources)", dest)
    )


def get_activation(name="silu", inplace=True):
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name == "lrelu":
        return nn.LeakyReLU(0.1, inplace=inplace)
    raise AttributeError("Unsupported act type: %s" % name)


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, ksize, stride, pad, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, act="silu"):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden, 1, 1, act=act)
        self.conv2 = BaseConv(hidden, out_channels, 3, 1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.use_add else y


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden, 1, 1, act=activation)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(ks, 1, ks // 2) for ks in kernel_sizes]
        )
        self.conv2 = BaseConv(hidden * (len(kernel_sizes) + 1), out_channels, 1, 1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.m], dim=1))


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, act="silu"):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden, 1, 1, act=act)
        self.conv2 = BaseConv(in_channels, hidden, 1, 1, act=act)
        self.conv3 = BaseConv(2 * hidden, out_channels, 1, 1, act=act)
        self.m = nn.Sequential(
            *[Bottleneck(hidden, hidden, shortcut, 1.0, act=act) for _ in range(n)]
        )

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), dim=1))


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        return self.conv(
            torch.cat(
                (
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ),
                dim=1,
            )
        )


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), act="silu"):
        super().__init__()
        self.out_features = out_features
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        self.stem = Focus(3, base_channels, ksize=3, act=act)
        self.dark2 = nn.Sequential(
            BaseConv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, act=act),
        )
        self.dark3 = nn.Sequential(
            BaseConv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, act=act),
        )
        self.dark4 = nn.Sequential(
            BaseConv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, act=act),
        )
        self.dark5 = nn.Sequential(
            BaseConv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, act=act
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {key: outputs[key] for key in self.out_features}


class YOLOPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"),
                 in_channels=(256, 512, 1024), act="silu"):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, act=act)
        self.in_features = in_features
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width), int(in_channels[1] * width),
            round(3 * depth), False, act=act,
        )
        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width), int(in_channels[0] * width),
            round(3 * depth), False, act=act,
        )
        self.bu_conv2 = BaseConv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width), int(in_channels[1] * width),
            round(3 * depth), False, act=act,
        )
        self.bu_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width), int(in_channels[2] * width),
            round(3 * depth), False, act=act,
        )

    def forward(self, x):
        features = self.backbone(x)
        x2, x1, x0 = [features[name] for name in self.in_features]

        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.C3_p4(torch.cat([self.upsample(fpn_out0), x1], 1))
        fpn_out1 = self.reduce_conv1(f_out0)
        pan_out2 = self.C3_p3(torch.cat([self.upsample(fpn_out1), x2], 1))
        pan_out1 = self.C3_n3(torch.cat([self.bu_conv2(pan_out2), fpn_out1], 1))
        pan_out0 = self.C3_n4(torch.cat([self.bu_conv1(pan_out1), fpn_out0], 1))
        return pan_out2, pan_out1, pan_out0


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, strides=(8, 16, 32),
                 in_channels=(256, 512, 1024), act="silu"):
        super().__init__()
        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True
        self.strides = strides
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        for channels in in_channels:
            hidden = int(256 * width)
            self.stems.append(BaseConv(int(channels * width), hidden, 1, 1, act=act))
            self.cls_convs.append(
                nn.Sequential(BaseConv(hidden, hidden, 3, 1, act=act), BaseConv(hidden, hidden, 3, 1, act=act))
            )
            self.reg_convs.append(
                nn.Sequential(BaseConv(hidden, hidden, 3, 1, act=act), BaseConv(hidden, hidden, 3, 1, act=act))
            )
            self.cls_preds.append(nn.Conv2d(hidden, self.n_anchors * num_classes, 1))
            self.reg_preds.append(nn.Conv2d(hidden, 4, 1))
            self.obj_preds.append(nn.Conv2d(hidden, self.n_anchors, 1))

    def forward(self, xin):
        outputs = []
        for k, x in enumerate(xin):
            x = self.stems[k](x)
            cls_output = self.cls_preds[k](self.cls_convs[k](x))
            reg_feat = self.reg_convs[k](x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            outputs.append(torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1))

        self.hw = [out.shape[-2:] for out in outputs]
        outputs = torch.cat([out.flatten(start_dim=2) for out in outputs], dim=2).permute(0, 2, 1)
        if self.decode_in_inference:
            return self.decode_outputs(outputs)
        return outputs

    def decode_outputs(self, outputs):
        device = outputs.device
        dtype = outputs.dtype
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid(
                torch.arange(hsize, device=device, dtype=dtype),
                torch.arange(wsize, device=device, dtype=dtype),
                indexing="ij",
            )
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            strides.append(torch.full((1, grid.shape[1], 1), stride, device=device, dtype=dtype))
        grids = torch.cat(grids, dim=1)
        strides = torch.cat(strides, dim=1)
        outputs = outputs.clone()
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs


class YOLOX(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


def build_yolox_x(num_classes=1, depth=1.33, width=1.25):
    in_channels = (256, 512, 1024)
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
    head = YOLOXHead(num_classes, width, in_channels=in_channels)
    return YOLOX(backbone, head)


def _strip_module_prefix(state):
    if not any(key.startswith("module.") for key in state):
        return state
    return {key[len("module."):]: value for key, value in state.items()}


def load_yolox_state(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
    state = _strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    return model


def preproc(image, input_size, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Letterbox + ImageNet normalize, matching ByteTrack ValTransform."""
    padded = np.ones((input_size[0], input_size[1], 3), dtype=np.float32) * 114.0
    img = np.asarray(image)
    ratio = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized = cv2.resize(
        img,
        (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    padded = padded[:, :, ::-1] / 255.0
    padded -= np.array(mean, dtype=np.float32)
    padded /= np.array(std, dtype=np.float32)
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), ratio


def postprocess(prediction, num_classes=1, conf_thre=0.2, nms_thre=0.7):
    """YOLOX decode is cxcywh. Return list of (N, 7) xyxy/obj/cls/class_id."""
    box = prediction.clone()
    box[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction = prediction.clone()
    prediction[:, :, :4] = box[:, :, :4]

    outputs = [None] * len(prediction)
    for i, image_pred in enumerate(prediction):
        if image_pred.size(0) == 0:
            continue
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        scores = image_pred[:, 4] * class_conf.squeeze(1)
        keep = scores >= conf_thre
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)[keep]
        if detections.size(0) == 0:
            continue
        nms_idx = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        outputs[i] = detections[nms_idx]
    return outputs


class YOLOXDetector:
    """Same detect(frame) -> [[x1,y1,x2,y2], ...] contract as YOLODetector."""

    def __init__(
        self,
        model_name="bytetrack-x",
        object_name="person",
        confidence_threshold=0.2,
        image_size=None,
        test_size=None,
        nms_threshold=0.7,
    ):
        self.object_name = object_name
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path, meta = resolve_yolox_checkpoint(model_name)
        if test_size is not None:
            self.test_size = (int(test_size[0]), int(test_size[1]))
        elif meta is not None:
            self.test_size = meta["test_size"]
        elif image_size:
            # Keep ByteTrack's 800:1440 aspect, honor a scalar long-side.
            long_side = int(image_size)
            self.test_size = (int(round(long_side * 800 / 1440 / 32) * 32), long_side)
        else:
            self.test_size = (800, 1440)

        print(
            "Loading YOLOX-X (%s)  test_size=%sx%s  conf=%.2f  nms=%.2f"
            % (os.path.basename(ckpt_path), self.test_size[0], self.test_size[1],
               self.confidence_threshold, self.nms_threshold),
            flush=True,
        )
        if meta and meta.get("leaks_mot_train"):
            print(
                "Note: %s was trained on %s. MOT16/17 train MOTA is not a clean hold-out."
                % (os.path.basename(ckpt_path), meta["train_data"]),
                flush=True,
            )

        self.model = build_yolox_x(num_classes=1)
        load_yolox_state(self.model, ckpt_path)
        self.model.to(self.device).eval()
        self.ckpt_path = ckpt_path

    @torch.no_grad()
    def detect(self, frame):
        img, ratio = preproc(frame, self.test_size)
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        outputs = self.model(tensor)
        dets = postprocess(
            outputs,
            num_classes=1,
            conf_thre=self.confidence_threshold,
            nms_thre=self.nms_threshold,
        )[0]
        if dets is None:
            return []

        boxes = dets[:, :4] / ratio
        bboxes = []
        for x1, y1, x2, y2 in boxes.cpu().numpy():
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x2 > x1 and y2 > y1:
                bboxes.append([x1, y1, x2, y2])
        return bboxes
