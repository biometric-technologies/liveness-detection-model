# Face liveness (presentation attack detection) models

Training and evaluation code for the face liveness check of the
[Biometric SDK](https://github.com/biometric-technologies/biometric-sdk).

## Models

- **MobileNetV2 classifier** ([Zawar, et al., 2023](https://www.atlantis-press.com/proceedings/acvait-22/125989871)),
  trained by `train.py`. Output: probability of an attack.
- **DeePixBiS** ([George and Marcel, 2019](https://arxiv.org/abs/1907.04047)), published as
  `deePix.tflite` in the [v0.2.0 release](https://github.com/biometric-technologies/liveness-detection-model/releases/tag/v0.2.0)
  and used by the SDK by default. It is not produced by the training code in this repository.
  Input: 1x3x224x224, RGB normalized to [-1, 1]. Output: a 14x14 map of live probabilities;
  the SDK uses its mean as the score (1.0 = live).

`test.py` evaluates both kinds and reports all scores as live scores.

## Requirements

- Python 3.9

## Install

Run `pip3 install -r requirements.txt`

## Dataset

- Structure:

    <pre>
    data
     ┬  
     ├ train
         ┬  
         ├ spoof  
         └ real 
     ├ validation  
         ┬  
         ├ spoof  
         └ real 
     └ test  
         ┬  
         ├ spoof  
         └ real
  </pre>

- If you have downloaded `CelebA` dataset -> run `python dataset.py` to prepare it for training:
  ```
  python3 dataset.py \
      --input_dir Data \
      --resized_dir data_resized \
      --target_size 224 \
  ```

## Run

- Run `python train.py` to initialize and train your model
  ```
  python3 train.py \
      --img_width 224 \
      --img_height 224 \
      --seed_number 24 \
      --train_batch_size 32 \
      --val_batch_size 32 \
      --batch_size 1 \
      --num_epochs 10 \
      --steps_per_epoch 100 \
      --validation_steps 1 \
      --input_data_path data_resized \
      --model_path model \
  ```
- Run `python tflite.py` to create your `.tflite` model
  ```
  python3 tflite.py \
      --saved_model_dir saved_mode_dir \
      --tflite_file_path model.tflite \
  ```
- Run `python test.py` to evaluate a model on the `test` folder
  ```
  python3 test.py \
      --threshold 0.5 \
      --test_dir test \
      --model_file model.tflite \
      --scores_file scores.csv
  ```
  Put bona fide images in `test/real` and attacks in `test/spoof`, optionally grouped by type
  (`test/spoof/print`, `test/spoof/replay`, ...). The script reports metrics in the terms of
  ISO/IEC 30107-3: APCER per attack type and for the worst type, BPCER, ACER, EER and BPCER at
  APCER 1%, 5% and 10%. Choose the SDK threshold from these numbers on images from your own
  cameras and attack instruments; a public dataset alone does not show how a model behaves in
  your deployment.

## Data

`dataset.py` prepares CelebA-Spoof. The validation set is split by subject (20% of subjects),
so the same person never appears in both training and validation.
- Optionally you can run `train_script.py` file to train number of models, convert them into `.tflite` files and test on images in some folder with number of thresholds.
  For example, if you run next command you'll train 4 models and test them on the list of thresholds:
  ```
  python3 train_script.py \
      --list_of_epochs 5 10 \
      --list_of_steps 10 100 \
      --list_of_thresholds 0.01 0.1 0.5 \
      --test_dir test
  ```

## License

Liveness detection model is released under the MIT License. See the [LICENSE](https://github.com/biometric-technologies/liveness-detection-model/blob/main/LICENSE.md) for details.