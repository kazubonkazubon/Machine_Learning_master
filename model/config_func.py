import configparser
import os
def reset_config():
    config = configparser.ConfigParser()
    config['no1'] = {
        'mid_units':128,
        'LEARNING_RATE':0.0001,
        'dropout_RATE':0.25,
        'dropout_RATE_2':0.5,
        'dropout_RATE_3':0.5,
        'epochs':100,
        'optimizer':"Adam",
    }
    config['no2'] = {
        'mid_units':512,
        'LEARNING_RATE':0.0001,
        'dropout_RATE':0.25,
        'dropout_RATE_2':0.25,
        'dropout_RATE_3':0.5,
        'epochs':100,
        'optimizer':"Adam",
    }

    config['no1_optimisation'] = {
        'Accuracy':0,
        'mid_units':128,
        'LEARNING_RATE':0.0001,
        'dropout_RATE':0.25,
        'dropout_RATE_2':0.5,
        'dropout_RATE_3':0.5,
        'epochs':100,
        'optimizer':"Adam",
        'activation':"relu"
    }
    config['no1_optimisation_with_train_dulation'] = {
        'Accuracy':0,
        'mid_units':128,
        'LEARNING_RATE':0.0001,
        'dropout_RATE':0.25,
        'dropout_RATE_2':0.5,
        'dropout_RATE_3':0.5,
        'epochs':100,
        'optimizer':"Adam",
        'activation':"relu"
    }
    config['no1_optimisation_with_train_val_dulation'] = {
        'Accuracy':0,
        'mid_units':128,
        'LEARNING_RATE':0.0001,
        'dropout_RATE':0.25,
        'dropout_RATE_2':0.5,
        'dropout_RATE_3':0.5,
        'epochs':100,
        'optimizer':"Adam",
        'activation':"relu"
    }
    config['no2_optimisation'] = {
        'Accuracy':0,
        'mid_units':512,
        'LEARNING_RATE':0.0001,
        'dropout_RATE':0.25,
        'dropout_RATE_2':0.25,
        'dropout_RATE_3':0.5,
        'epochs':100,
        'optimizer':"Adam",
        'activation':"relu"
    }
    config['no2_optimisation_with_train_dulation'] = {
        'Accuracy':0,
        'mid_units':512,
        'LEARNING_RATE':0.0001,
        'dropout_RATE':0.25,
        'dropout_RATE_2':0.25,
        'dropout_RATE_3':0.5,
        'epochs':100,
        'optimizer':"Adam",
        'activation':"relu"
    }
    config['no2_optimisation_with_train_val_dulation'] = {
        'Accuracy':0,
        'mid_units':512,
        'LEARNING_RATE':0.0001,
        'dropout_RATE':0.25,
        'dropout_RATE_2':0.25,
        'dropout_RATE_3':0.5,
        'epochs':100,
        'optimizer':"Adam",
        'activation':"relu"
    }
    config['accuracy'] = {
        'ACCURACY':0
    }
    ini_file_path="/content/drive/MyDrive/parameters/config.ini"

    if os.path.exists(ini_file_path):
        os.remove(ini_file_path)

    with open(ini_file_path, 'x') as file:
        config.write(file)