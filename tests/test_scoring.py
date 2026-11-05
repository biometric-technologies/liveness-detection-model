import cv2
import numpy as np
import pytest
import tensorflow as tf

from conftest import load_script

evaluation = load_script("liveness_evaluation", "test.py")


def tflite(model, path):
    path.write_bytes(tf.lite.TFLiteConverter.from_keras_model(model).convert())
    return str(path)


def test_classifier_output_is_turned_into_a_live_score(tmp_path):
    # Stands in for the MobileNetV2 classifier of train.py: NHWC input and one
    # sigmoid output, P(spoof), here fixed at 0.8.
    inputs = tf.keras.Input((16, 16, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    out = tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer="zeros",
                                bias_initializer=tf.keras.initializers.Constant(np.log(0.8 / 0.2)))(x)
    model = evaluation.LivenessModel(tflite(tf.keras.Model(inputs, out), tmp_path / "classifier.tflite"))
    assert not model.nchw and (model.height, model.width) == (16, 16)

    image = tmp_path / "face.jpg"
    cv2.imwrite(str(image), np.full((40, 30, 3), 128, np.uint8))
    assert model.live_score(str(image)) == pytest.approx(0.2, abs=1e-5)


def test_deepixbis_map_is_averaged_after_its_normalization(tmp_path):
    # Stands in for DeePixBiS: NCHW input normalized to [-1, 1], a map of P(live).
    # This one returns its input, so the score is the mean normalized pixel.
    inputs = tf.keras.Input((3, 8, 8))
    model = evaluation.LivenessModel(tflite(tf.keras.Model(inputs, tf.identity(inputs)), tmp_path / "map.tflite"))
    assert model.nchw and (model.height, model.width) == (8, 8)

    white = tmp_path / "white.png"
    cv2.imwrite(str(white), np.full((20, 20, 3), 255, np.uint8))
    assert model.live_score(str(white)) == pytest.approx(1.0, abs=1e-5)
    black = tmp_path / "black.png"
    cv2.imwrite(str(black), np.zeros((20, 20, 3), np.uint8))
    assert model.live_score(str(black)) == pytest.approx(-1.0, abs=1e-5)


def test_images_lists_only_image_files_in_order(tmp_path):
    for name in ("b.jpg", "a.PNG", "notes.txt", "c.bmp"):
        (tmp_path / name).write_bytes(b"")
    assert [p.split("/")[-1] for p in evaluation.images(str(tmp_path))] == ["a.PNG", "b.jpg", "c.bmp"]
