import os
import subprocess
import sys
import zlib

import cv2
import numpy as np

from conftest import ROOT
from dataset import resize_image_with_aspect_ratio


def test_resize_pads_to_a_square_without_swapping_colours():
    blue = np.zeros((40, 80, 3), np.uint8)
    blue[..., 0] = 255  # OpenCV images are BGR
    wide = resize_image_with_aspect_ratio(blue, 32)
    assert wide.shape == (32, 32, 3)
    # 80x40 becomes 32x16, centred: 8 rows of padding above and below
    assert (wide[:8] == 0).all() and (wide[-8:] == 0).all()
    assert (wide[8:24, :, 0] == 255).all() and (wide[8:24, :, 1:] == 0).all()

    tall = resize_image_with_aspect_ratio(blue.transpose(1, 0, 2).copy(), 32)
    assert tall.shape == (32, 32, 3)
    assert (tall[:, :8] == 0).all() and (tall[:, 8:24, 0] == 255).all()


def make_celeba_spoof(root, subjects):
    """CelebA-Spoof layout: <split>/<subject>/{live,spoof}/*.jpg."""
    rng = np.random.default_rng(0)
    for split, names in subjects.items():
        for subject in names:
            for kind, count in (("live", 2), ("spoof", 3)):
                folder = os.path.join(root, split, subject, kind)
                os.makedirs(folder)
                for i in range(count):
                    image = rng.integers(0, 256, (60, 40, 3), dtype=np.uint8)
                    cv2.imwrite(os.path.join(folder, f"{i:03d}.jpg"), image)


def subjects_in(folder):
    return {name.split("_")[0] for name in os.listdir(folder)}


def test_split_keeps_every_subject_in_one_split(tmp_path):
    train = [f"{n}" for n in range(1000, 1030)]
    test = [f"{n}" for n in range(2000, 2005)]
    data, resized = str(tmp_path / "data"), str(tmp_path / "resized")
    make_celeba_spoof(data, {"train": train, "test": test})

    subprocess.run([sys.executable, os.path.join(ROOT, "dataset.py"), "-i", data, "-r", resized, "-s", "32"],
                   check=True, cwd=str(tmp_path))

    to_valid = {s for s in train if zlib.crc32(s.encode()) % 5 == 0}
    assert to_valid and to_valid != set(train)
    for label, per_subject in (("real", 2), ("spoof", 3)):
        train_subjects = subjects_in(os.path.join(resized, "train", label))
        valid_subjects = subjects_in(os.path.join(resized, "valid", label))
        assert valid_subjects == to_valid
        assert train_subjects == set(train) - to_valid
        assert subjects_in(os.path.join(resized, "test", label)) == set(test)
        assert len(os.listdir(os.path.join(resized, "valid", label))) == per_subject * len(to_valid)
        for split in ("train", "valid", "test"):
            folder = os.path.join(resized, split, label)
            for name in os.listdir(folder):
                assert cv2.imread(os.path.join(folder, name)).shape == (32, 32, 3)
