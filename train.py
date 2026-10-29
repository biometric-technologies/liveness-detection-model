import os
import glob
import pandas as pd
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from model import GetModel
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--img_width", type=int, default=224)
parser.add_argument("-ht", "--img_height", type=int, default=224)
parser.add_argument("-sd", "--seed_number", type=int, default=24)
parser.add_argument("-tbs", "--train_batch_size", type=int, default=32)
parser.add_argument("-vbs", "--val_batch_size", type=int, default=32)
parser.add_argument("-bs", "--batch_size", type=int, default=1)
parser.add_argument("-e", "--num_epochs", type=int, default=30)
parser.add_argument("-s", "--steps_per_epoch", type=int, default=None, help="default: the whole training set")
parser.add_argument("-vs", "--validation_steps", type=int, default=None, help="default: the whole validation set")
parser.add_argument("-i", "--input_data_path", type=str, default='data_resized', help="path to root 'data' folder")
parser.add_argument("-m", "--model_path", type=str, default='model', help="path to model folder")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Limit GPU memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
    args = vars(parser.parse_args())

    img_width = args["img_width"]
    img_height = args["img_height"]

    seed_number = args["seed_number"]

    train_batch_size = args["train_batch_size"]
    val_batch_size = args["val_batch_size"]
    batch_size = args["batch_size"]

    num_epochs = args["num_epochs"]
    steps_per_epoch = args["steps_per_epoch"]
    validation_steps = args["validation_steps"]

    model_dir = args["model_path"]

    model = GetModel(img_width, img_height)

    input_dir = args["input_data_path"]
    train_dir = os.path.join(input_dir, 'train')
    val_dir = os.path.join(input_dir, 'valid')
    test_dir = os.path.join(input_dir, 'test')
    label_name = [subdir for subdir in sorted(os.listdir(train_dir)) if os.path.isdir(os.path.join(train_dir, subdir))]
    dir_dict = {'train': train_dir, 'val': val_dir, 'test': test_dir}
    case_count, set_length = {}, {}

    for key, val in dir_dict.items():
        case_count[key] = {}
        set_count = 0

        for label in label_name:
            label_list = list(sorted(glob.glob(os.path.join(val, label, "*.png"))))
            if len(label_list) == 0:
                label_list = list(sorted(glob.glob(os.path.join(val, label, "*.jpg"))))

            case_count[key][label] = len(label_list)
            set_count += len(label_list)

        set_length[key] = set_count

    case_count_df = pd.DataFrame(case_count)

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.15,
                                       zoom_range=0.15,
                                       horizontal_flip=True,
                                       fill_mode="nearest")
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_directory(train_dir,
                                                  batch_size=train_batch_size,
                                                  class_mode='binary',
                                                  target_size=(img_width, img_height),
                                                  seed=seed_number)

    val_gen = val_datagen.flow_from_directory(val_dir,
                                              batch_size=val_batch_size,
                                              class_mode='binary',
                                              target_size=(img_width, img_height),
                                              shuffle=False)

    train_length = len(train_gen.classes)
    weight0 = train_length / case_count_df['train'][label_name[0]] * (1 / len(label_name))
    weight1 = train_length / case_count_df['train'][label_name[1]] * (1 / len(label_name))
    class_weight = {0: weight0, 1: weight1}

    train_id = "train-weights"
    save_dir = os.path.join(model_dir, train_id)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    cont_filepath = "mobilenetv2-epoch_{epoch:02d}.hdf5"
    cont_checkpoint = ModelCheckpoint(os.path.join(save_dir, cont_filepath))

    best_filepath = "mobilenetv2-best.hdf5"
    best_checkpoint = ModelCheckpoint(os.path.join(save_dir, best_filepath), monitor='val_auc', mode='max',
                                      save_best_only=True, save_weights_only=True)

    plateau_scheduler = ReduceLROnPlateau(factor=0.2, patience=3, verbose=1, min_delta=0.005, min_lr=5e-7)
    # stop when validation AUC stops improving and keep the best epoch, not the last one
    early_stopping = EarlyStopping(monitor='val_auc', mode='max', patience=6, restore_best_weights=True)

    history = model.fit(train_gen,
                        epochs=num_epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_gen,
                        validation_steps=validation_steps,
                        callbacks=[best_checkpoint, cont_checkpoint, plateau_scheduler, early_stopping],
                        class_weight=class_weight)

    saved_model_path = os.path.join(model_dir, 'saved_model_dir')
    tf.keras.models.save_model(model, saved_model_path)

    # Plotting training loss
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    img_path = os.path.join(model_dir, 'train_metrics.png')

    plt.savefig(img_path, dpi=300, bbox_inches='tight')
