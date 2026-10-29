import os
import subprocess
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("-el", "--list_of_epochs", type=int, nargs='+', default=[1])
parser.add_argument("-sl", "--list_of_steps", type=int, nargs='+', default=[1])
parser.add_argument("-tl", "--list_of_thresholds", type=float, nargs='+', default=[0.5])
parser.add_argument("-td", "--test_dir", type=str, default=None, help="path to 'test' folder")


if __name__ == '__main__':
    args = vars(parser.parse_args())

    list_of_epochs = args["list_of_epochs"]
    list_of_steps = args["list_of_steps"]
    list_of_thresholds = args["list_of_thresholds"]
    test_dir = args["test_dir"]

    if not os.path.isdir('models'):
        os.makedirs('models')

    outputfile_path = os.path.join('models', "result.csv")
    if not os.path.exists(outputfile_path):
        with open(outputfile_path, mode="a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'steps_per_epoch', 'threshold', 'apcer', 'bpcer', 'acer', 'eer'])

    for epoch in list_of_epochs:
        for step in list_of_steps:
            model_path = f"models/model_{epoch}_{step}"

            try:
                subprocess.run(["python", "train.py"] + ["--num_epochs", str(epoch), "--steps_per_epoch", str(step), "--model_path", model_path], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running train.py: {e}")

            saved_model_dir = os.path.join(model_path, 'saved_model_dir')
            tflite_path = os.path.join(model_path, 'model.tflite')
            if not os.path.exists(tflite_path):
                try:
                    subprocess.run(
                        ["python", "tflite.py"] + ["--saved_model_dir", saved_model_dir, "--tflite_file_path", tflite_path], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error running tflite.py: {e}")

            for threshold in list_of_thresholds:
                try:
                    subprocess.run(
                        ["python", "test.py"] + ["--threshold", str(threshold), "--test_dir", test_dir, "--model_file", tflite_path, "--output_file", outputfile_path, "--epoch", str(epoch), "--steps_per_epoch", str(step)], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error running test.py: {e}")
