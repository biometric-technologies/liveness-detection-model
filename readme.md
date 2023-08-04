# Liveness face detection tensorflow lite model

Model was taken from [Face Anti-Spoofing Detection using MobileNetV2](https://www.kaggle.com/code/faber24/face-anti-spoofing-detection-using-mobilenetv2/notebook)

## Requirements

- Python 3.9

## Install

Run `pip3 install -r requirements.txt`

## Run

- Set up 2 images directory with folders inside

    <pre>data folder
     ┬
     ├ data
         ┬  
         ├ training
             ┬  
             ├ spoof  
             └ real 
         ├ evaluation  
             ┬  
             ├ spoof  
             └ real 
         └ development  
             ┬  
             ├ spoof  
             └ real
  
    test folder
        ┬
        ├ test
          ┬  
          ├ spoof  
          └ real
  </pre>

- Run `python train.py` to initialize and train your model
- Run `python tflite.py` to create your `.tflite` model
- Run `python test.py` to test your model on images in `test` folder