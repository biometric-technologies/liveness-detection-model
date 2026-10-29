"""Evaluates a liveness .tflite model on a test folder (ISO/IEC 30107-3 metrics).

Folder layout:
    test/real/*.jpg                    bona fide presentations
    test/spoof/*.jpg                   attacks, or
    test/spoof/<attack type>/*.jpg     attacks grouped by type (print, replay, mask...)

Two model kinds are supported and detected from the input shape:
    * MobileNetV2 from train.py: NHWC input, RGB / 255, one sigmoid output = P(spoof)
    * DeePixBiS (deePix.tflite used by the SDK): NCHW input, (RGB / 255 - 0.5) / 0.5,
      14x14 map of P(live); the score is the mean of the map
All scores are converted to live scores (higher = bona fide).
"""
import argparse
import csv
import os

import cv2
import numpy as np

from pad_metrics import apcer, bpcer, bpcer_at_apcer, eer, worst_apcer

try:
    from tensorflow.lite import Interpreter
except ImportError:  # lightweight runtime
    from ai_edge_litert.interpreter import Interpreter

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--threshold", type=float, default=0.5, help="live-score threshold")
parser.add_argument("-d", "--test_dir", type=str, default='test', help="path to 'test' folder")
parser.add_argument("-m", "--model_file", type=str, default='model.tflite', help="path to '.tflite' model file")
parser.add_argument("-f", "--output_file", type=str, default='result.csv', help="CSV to append the summary to")
parser.add_argument("-sf", "--scores_file", type=str, default=None, help="optional CSV with the score of every image")
parser.add_argument("-e", "--epoch", type=int, default=1)
parser.add_argument("-s", "--steps_per_epoch", type=int, default=1)


class LivenessModel:
    def __init__(self, path):
        self.interpreter = Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        shape = self.input['shape']
        self.nchw = shape[1] == 3
        self.height, self.width = (shape[2], shape[3]) if self.nchw else (shape[1], shape[2])

    def live_score(self, image_path):
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(cv2.resize(bgr, (self.width, self.height)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        if self.nchw:
            x = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None]
        else:
            x = rgb[None]
        self.interpreter.set_tensor(self.input['index'], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output['index'])
        value = float(np.mean(out))
        # DeePixBiS predicts "live"; the MobileNetV2 classifier predicts "spoof" (class 1 of real/spoof)
        return value if self.nchw else 1.0 - value


def images(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXT))


def main():
    args = vars(parser.parse_args())
    if not os.path.isfile(args["model_file"]):
        print(f"There is no {args['model_file']} file")
        return
    model = LivenessModel(args["model_file"])
    threshold = args["threshold"]
    real_dir = os.path.join(args["test_dir"], "real")
    spoof_dir = os.path.join(args["test_dir"], "spoof")

    rows = []
    bona_fide = []
    for path in images(real_dir):
        s = model.live_score(path)
        bona_fide.append(s)
        rows.append((path, "bona_fide", "", s))

    attacks = {}
    groups = [d for d in sorted(os.listdir(spoof_dir)) if os.path.isdir(os.path.join(spoof_dir, d))]
    for attack_type in groups or ["attack"]:
        folder = os.path.join(spoof_dir, attack_type) if groups else spoof_dir
        for path in images(folder):
            s = model.live_score(path)
            attacks.setdefault(attack_type, []).append(s)
            rows.append((path, "attack", attack_type, s))

    all_attacks = np.concatenate([np.asarray(v) for v in attacks.values()])
    worst, worst_type = worst_apcer(attacks, threshold)
    b = bpcer(bona_fide, threshold)
    e, e_threshold = eer(bona_fide, all_attacks)
    print(f"Bona fide: {len(bona_fide)}, attacks: {len(all_attacks)} in {len(attacks)} type(s)")
    for attack_type, scores in attacks.items():
        print(f"APCER {attack_type}: {apcer(scores, threshold) * 100:.2f}%")
    print(f"APCER (worst type, {worst_type}): {worst * 100:.2f}%")
    print(f"BPCER: {b * 100:.2f}%")
    print(f"ACER: {(worst + b) / 2 * 100:.2f}%")
    print(f"EER: {e * 100:.2f}% at threshold {e_threshold:.3f}")
    for target in (0.01, 0.05, 0.10):
        value, t = bpcer_at_apcer(bona_fide, attacks, target)
        print(f"BPCER at APCER {target * 100:.0f}%: {value * 100:.2f}% (threshold {t:.3f})")

    with open(args["output_file"], mode="a", newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(['epoch', 'steps_per_epoch', 'threshold', 'apcer', 'bpcer', 'acer', 'eer'])
        writer.writerow([args["epoch"], args["steps_per_epoch"], threshold, worst, b, (worst + b) / 2, e])
    if args["scores_file"]:
        with open(args["scores_file"], mode="w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'presentation', 'attack_type', 'live_score'])
            writer.writerows(rows)


if __name__ == '__main__':
    main()
