import numpy as np
from keras.applications import mobilenet_v2

import model


def test_classifier_head(monkeypatch):
    # The real model starts from ImageNet weights; random weights are enough
    # to check the architecture and do not need a download.
    backbone = mobilenet_v2.MobileNetV2
    monkeypatch.setattr(model.mobilenet_v2, "MobileNetV2",
                        lambda **kwargs: backbone(**{**kwargs, "weights": None}))
    m = model.GetModel(96, 96)
    assert m.output_shape == (None, 1)
    assert m.layers[-1].name == "classifier"
    assert all(layer.trainable for layer in m.layers)
    out = m.predict(np.random.default_rng(0).random((2, 96, 96, 3), dtype=np.float32), verbose=0)
    assert ((out > 0) & (out < 1)).all()
    assert m.loss == "binary_crossentropy"
