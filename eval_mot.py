"""Evaluate CLIPSORT / DeepSORT on MOT Challenge benchmarks (MOT16, MOT17, MOT20).

Download the train split from https://motchallenge.net (registration required):

    MOT16/train/MOT16-02/img1, gt/gt.txt, det/det.txt, seqinfo.ini
    MOT17/train/MOT17-02-FRCNN/...
    MOT20/train/MOT20-01/...

Examples:

    # MOT17, private YOLO detections (one copy of each video)
    python eval_mot.py --data-root /path/to/MOT17 --split train

    # Official public-detection protocol (all detector variants)
    python eval_mot.py --data-root /path/to/MOT17 --split train --public-dets

    # Smoke test one short sequence
    python eval_mot.py --data-root /path/to/MOT17 --seqs MOT17-05 --max-frames 50

Requires: pip install motmetrics
"""

import argparse
import os
import sys
import time

import yaml

def make_pbar(iterable=None, **kwargs):
    """tqdm progress bar, with a plain-text fallback if tqdm is missing."""
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("smoothing", 0.08)
    kwargs.setdefault("mininterval", 0.3)
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs) if iterable is not None else tqdm(**kwargs)
    except ImportError:
        return _FallbackBar(iterable, **kwargs)


def pbar_write(message):
    try:
        from tqdm import tqdm
        tqdm.write(message)
    except ImportError:
        print(message)


class _FallbackBar:
    def __init__(self, iterable=None, total=None, desc="", unit="it", **kwargs):
        self.iterable = iterable
        self.total = total if total is not None else (len(iterable) if iterable is not None else 0)
        self.desc = desc
        self.unit = unit
        self.n = 0
        self._last_print = 0.0

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.update(1)
        self.close()

    def update(self, n=1):
        self.n += n
        now = time.time()
        if self.n >= self.total or now - self._last_print >= 2.0:
            self._last_print = now
            print("\r%s: %d/%d %s" % (self.desc, self.n, self.total, self.unit), end="", flush=True)

    def set_postfix(self, **kwargs):
        return None

    def close(self):
        if self.total:
            print("\r%s: %d/%d %s" % (self.desc, self.n, self.total, self.unit), flush=True)


# MOT17 ships the same 7 videos three times (DPM / FRCNN / SDP).
MOT17_DETECTOR_PRIORITY = ("FRCNN", "SDP", "DPM")


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate this tracker on MOT16 / MOT17 / MOT20."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to MOT16, MOT17, or MOT20 (or a single sequence folder).",
    )
    parser.add_argument("--split", default="train", help="Dataset split (train/test).")
    parser.add_argument(
        "--seqs",
        nargs="+",
        default=None,
        help="Sequence names or prefixes, e.g. MOT17-05 MOT17-02-FRCNN.",
    )
    parser.add_argument("--config", default="config.yaml", help="Tracker YAML config.")
    parser.add_argument(
        "--output-dir",
        default="./results/mot_eval",
        help="Directory for MOT result files and the metrics table.",
    )
    parser.add_argument(
        "--public-dets",
        action="store_true",
        help="Use official det/det.txt instead of YOLO (MOTChallenge protocol).",
    )
    parser.add_argument(
        "--class-name",
        default="person",
        help="Detector class to keep for Ultralytics backends (ignored by YOLOX).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on frames per sequence (debugging).",
    )
    parser.add_argument(
        "--skip-track",
        action="store_true",
        help="Only evaluate existing result files in --output-dir.",
    )
    return parser.parse_args()


def is_sequence_dir(path):
    return os.path.isdir(os.path.join(path, "img1"))


def sequence_base_name(name):
    """MOT17-02-FRCNN -> MOT17-02; other names are unchanged."""
    parts = name.split("-")
    if len(parts) >= 3 and parts[0].upper().startswith("MOT") and parts[2] in MOT17_DETECTOR_PRIORITY:
        return "-".join(parts[:2])
    return name


def prefer_mot17_variant(names):
    """Keep one folder per video when using private detections."""
    grouped = {}
    for name in names:
        grouped.setdefault(sequence_base_name(name), []).append(name)

    chosen = []
    for base, variants in grouped.items():
        if len(variants) == 1:
            chosen.append(variants[0])
            continue
        picked = None
        for detector in MOT17_DETECTOR_PRIORITY:
            match = next((v for v in variants if v.endswith("-" + detector)), None)
            if match:
                picked = match
                break
        chosen.append(picked or sorted(variants)[0])
    return sorted(chosen)


def discover_sequences(data_root, split, requested, use_public_dets):
    """Find MOT sequence folders under data_root."""
    if is_sequence_dir(data_root):
        candidates = [data_root]
    else:
        split_dir = os.path.join(data_root, split)
        search_dir = split_dir if os.path.isdir(split_dir) else data_root
        if not os.path.isdir(search_dir):
            raise FileNotFoundError("No sequences found under %s" % data_root)
        names = sorted(
            name for name in os.listdir(search_dir)
            if is_sequence_dir(os.path.join(search_dir, name))
        )
        if requested:
            selected = []
            for req in requested:
                matches = [name for name in names if name == req or name.startswith(req)]
                if not matches:
                    raise ValueError("No sequence matching '%s' in %s" % (req, search_dir))
                selected.extend(matches)
            names = sorted(set(selected))
        if not use_public_dets:
            names = prefer_mot17_variant(names)
        candidates = [os.path.join(search_dir, name) for name in names]

    if not candidates:
        raise FileNotFoundError("No MOT sequences with an img1/ folder under %s" % data_root)
    return candidates


def list_frame_files(seq_dir, max_frames=None):
    img_dir = os.path.join(seq_dir, "img1")
    frames = sorted(
        name for name in os.listdir(img_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


def load_public_detections(det_path, min_conf):
    """Load MOT det.txt into {frame_id: [[x1, y1, x2, y2], ...]}."""
    detections = {}
    if not os.path.isfile(det_path):
        raise FileNotFoundError("Public detections not found: %s" % det_path)

    with open(det_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            frame_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            if conf < min_conf or w <= 0 or h <= 0:
                continue
            detections.setdefault(frame_id, []).append(
                [int(x), int(y), int(x + w), int(y + h)]
            )
    return detections


def clip_boxes(bboxes, width, height):
    clipped = []
    for x1, y1, x2, y2 in bboxes:
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(0, min(width, int(x2)))
        y2 = max(0, min(height, int(y2)))
        if x2 > x1 and y2 > y1:
            clipped.append([x1, y1, x2, y2])
    return clipped


def write_mot_results(path, results):
    """Write MOT Challenge tracker output: frame,id,x,y,w,h,conf,-1,-1,-1."""
    with open(path, "w") as handle:
        for frame_id, track_id, x, y, w, h, conf in results:
            handle.write(
                "%d,%d,%.2f,%.2f,%.2f,%.2f,%.6f,-1,-1,-1\n"
                % (frame_id, track_id, x, y, w, h, conf)
            )


def build_detector(config, class_name, use_public_dets):
    if use_public_dets:
        return None
    from yolo import build_detector as _build_detector

    params = config["params"]
    return _build_detector(
        params["detection_backbone"],
        class_name,
        params["detection_confidence_threshold"],
        params.get("detection_imgsz", 640),
        test_size=params.get("detection_test_size"),
    )


def build_tracker(config):
    from deepsort import DeepSort

    params = config["params"]
    return DeepSort(
        params["max_age"],
        params["min_hits"],
        params["feature_extractor"],
        params["mahalanobis_threshold"],
        params["cosine_threshold"],
        params["iou_threshold"],
    )


def run_sequence(seq_dir, detector, tracker, min_conf, use_public_dets, max_frames,
                 overall_bar=None, seq_index=1, n_seqs=1):
    """Run the existing tracker on one MOT sequence and return MOT rows."""
    import cv2

    tracker.reset()
    frame_files = list_frame_files(seq_dir, max_frames)
    img_dir = os.path.join(seq_dir, "img1")

    public_dets = None
    if use_public_dets:
        public_dets = load_public_detections(
            os.path.join(seq_dir, "det", "det.txt"), min_conf
        )

    results = []
    seq_name = os.path.basename(seq_dir.rstrip(os.sep))
    seq_bar = make_pbar(
        frame_files,
        desc="[%d/%d] %s" % (seq_index, n_seqs, seq_name),
        unit="frm",
        leave=False,
        position=1,
    )
    for frame_file in seq_bar:
        frame_id = int(os.path.splitext(frame_file)[0])
        frame = cv2.imread(os.path.join(img_dir, frame_file))
        if frame is None:
            if overall_bar is not None:
                overall_bar.update(1)
            continue

        height, width = frame.shape[:2]
        if public_dets is not None:
            bboxes = public_dets.get(frame_id, [])
        else:
            bboxes = detector.detect(frame)
        bboxes = clip_boxes(bboxes, width, height)

        tracks = tracker.update(bboxes, frame)
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            results.append((frame_id, track.track_id, x1, y1, x2 - x1, y2 - y1, 1.0))

        seq_bar.set_postfix(dets=len(bboxes), tracks=len(tracks), ids=tracker.next_id - 1)
        if overall_bar is not None:
            overall_bar.update(1)
            overall_bar.set_postfix(seq=seq_name)
    seq_bar.close()
    return results, len(frame_files)


def _import_motmetrics():
    """Load motmetrics, including a NumPy 2.x compatibility shim."""
    import numpy as np

    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

    import motmetrics as mm
    return mm


def load_ground_truth(gt_path):
    mm = _import_motmetrics()

    with open(gt_path, "r") as handle:
        first = handle.readline().strip()
    n_cols = len(first.split(",")) if first else 0

    # mot16 and mot15-2D share a loader. MOT16/17/20 GT has 9 columns and a
    # real class label (1 = pedestrian); MOT15 has 10 columns and no class.
    gt = mm.io.loadtxt(gt_path, fmt="mot16", min_confidence=1)
    if n_cols == 9 and "ClassId" in gt.columns:
        gt = gt[gt["ClassId"] == 1].copy()
    return gt


def evaluate_sequences(eval_items, output_dir):
    """Score tracker files against MOT GT with motmetrics (MOTA, IDF1, ...)."""
    try:
        mm = _import_motmetrics()
    except ImportError:
        print("Install motmetrics to compute scores:  pip install motmetrics")
        return None

    accs = []
    names = []
    score_bar = make_pbar(eval_items, desc="Scoring", unit="seq", leave=False)
    for seq_name, gt_path, result_path in score_bar:
        if not os.path.isfile(gt_path):
            pbar_write("Skipping %s (no ground truth)." % seq_name)
            continue
        if not os.path.isfile(result_path):
            pbar_write("Skipping %s (no result file)." % seq_name)
            continue

        gt = load_ground_truth(gt_path)
        ts = mm.io.loadtxt(result_path, fmt="mot15-2D", min_confidence=-1.0)
        accs.append(mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5))
        names.append(seq_name)

    if not accs:
        print("No sequences with both results and ground truth.")
        return None

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=True,
    )
    table = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
    print("\nMOT Challenge metrics (IoU=0.5, pedestrians only)")
    print(table)

    csv_path = os.path.join(output_dir, "metrics.csv")
    summary.to_csv(csv_path)
    txt_path = os.path.join(output_dir, "metrics.txt")
    with open(txt_path, "w") as handle:
        handle.write(table + "\n")
    print("Saved metrics to %s and %s" % (csv_path, txt_path))
    return summary


def save_run_config(output_dir, args, config):
    snapshot = {
        "cli": vars(args),
        "params": config.get("params", {}),
    }
    with open(os.path.join(output_dir, "run_config.yaml"), "w") as handle:
        yaml.safe_dump(snapshot, handle, sort_keys=False)


def main():
    args = parse_args()
    if not os.path.isfile(args.config):
        print("Config not found: %s" % args.config)
        sys.exit(1)

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    save_run_config(args.output_dir, args, config)

    sequences = discover_sequences(
        args.data_root, args.split, args.seqs, args.public_dets
    )
    print("Sequences (%d): %s" % (
        len(sequences),
        ", ".join(os.path.basename(path) for path in sequences),
    ))
    print("Feature extractor: %s" % config["params"]["feature_extractor"])
    print("Detections: %s" % (
        "public MOT" if args.public_dets
        else "%s / %s" % (config["params"]["detection_backbone"], args.class_name)
    ))

    detector = None
    tracker = None
    if not args.skip_track:
        if not args.public_dets:
            print("Loading detector (%s, imgsz=%s)..." % (
                config["params"]["detection_backbone"],
                config["params"].get("detection_test_size")
                or config["params"].get("detection_imgsz", 640),
            ))
        detector = build_detector(config, args.class_name, args.public_dets)
        print("Loading %s feature extractor..." % config["params"]["feature_extractor"])
        tracker = build_tracker(config)
        print("Models ready.\n")

    eval_items = []
    start = time.time()
    overall_bar = None
    if not args.skip_track:
        total_frames = sum(
            len(list_frame_files(seq_dir, args.max_frames)) for seq_dir in sequences
        )
        overall_bar = make_pbar(
            total=total_frames,
            desc="Overall",
            unit="frm",
            position=0,
            leave=True,
        )

    n_seqs = len(sequences)
    for seq_index, seq_dir in enumerate(sequences, 1):
        seq_name = os.path.basename(seq_dir.rstrip(os.sep))
        result_path = os.path.join(args.output_dir, seq_name + ".txt")
        gt_path = os.path.join(seq_dir, "gt", "gt.txt")
        eval_items.append((seq_name, gt_path, result_path))

        if args.skip_track:
            continue

        seq_start = time.time()
        results, n_frames = run_sequence(
            seq_dir,
            detector,
            tracker,
            config["params"]["detection_confidence_threshold"],
            args.public_dets,
            args.max_frames,
            overall_bar=overall_bar,
            seq_index=seq_index,
            n_seqs=n_seqs,
        )
        write_mot_results(result_path, results)
        unique_ids = len({row[1] for row in results})
        pbar_write(
            "  finished %s  frames=%d  boxes=%d  ids=%d  %.1fs"
            % (seq_name, n_frames, len(results), unique_ids, time.time() - seq_start)
        )

    if overall_bar is not None:
        overall_bar.close()

    if not args.skip_track:
        elapsed = time.time() - start
        print("\nTracking time: %.1f min (%.2f frm/s)" % (
            elapsed / 60.0,
            (overall_bar.n / elapsed) if elapsed > 0 else 0.0,
        ))

    if args.split == "test":
        print("Test split has no public GT. Submit the .txt files on motchallenge.net.")
        return

    evaluate_sequences(eval_items, args.output_dir)


if __name__ == "__main__":
    main()
