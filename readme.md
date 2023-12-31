# Model

- MobileNetV2 ([Zawar, et al., 2023](https://www.atlantis-press.com/proceedings/acvait-22/125989871)) 

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
- Run `python test.py` to test your model on images in `test` folder
  ```
  python3 test.py \
      --threshold 0.5 \
      --test_dir test \
      --model_file model.tflite
  ```
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