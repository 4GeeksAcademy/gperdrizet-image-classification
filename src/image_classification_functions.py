'''Helper function for training of convolutional neural network classifier.'''

# Handle imports up-front
import os
import itertools
import pickle
import zipfile
import shutil
import glob
import random
from typing import Tuple
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# Figure out if we are in a Kaggle notebook or not
working_directory=os.getcwd()
path_list=working_directory.split(os.sep)
path_list=[s for s in path_list if s]

if len(path_list) >= 1:
    if path_list[0] == 'kaggle':
        environment='kaggle'

    else:
        environment='other'
else:
    environment='other'

# Set experiment save file directory accordingly
if environment == 'kaggle':
    EXPERIMENT_DATA_PATH='/kaggle/working/experiment_results'

else:
    EXPERIMENT_DATA_PATH='../data/experiment_results'

# Make the experiment save file directory
Path(EXPERIMENT_DATA_PATH).mkdir(parents=True, exist_ok=True)

####################################################################
# Convolutional neural network training and optimization functions #
####################################################################

# Set some global default values for how long/how much to train
BATCH_SIZE=8
LEARNING_RATE=0.1
L1_PENALTY=None
L2_PENALTY=None
IMAGE_HEIGHT=64
IMAGE_WIDTH=48
ASPECT_RATIO=4/3
FILTER_NUMS=[16,32,64]
FILTER_SIZE=3

SINGLE_TRAINING_RUN_EPOCHS=20
OPTIMIZATION_TRAINING_RUN_EPOCHS=10
STEPS_PER_EPOCH=100
VALIDATION_STEPS=100

# Define a re-usable helper function to create training and validation datasets
def make_datasets(
        training_data_path: str,
        validation_data_path: str,
        image_width: int=IMAGE_WIDTH,
        image_height: int=IMAGE_HEIGHT,
        batch_size: int=BATCH_SIZE
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    '''Makes training and validation dataset generator objects.'''

    training_dataset=tf.keras.utils.image_dataset_from_directory(
        training_data_path,
        image_size=(image_height, image_width),
        color_mode='grayscale',
        batch_size=batch_size
    )

    validation_dataset=tf.keras.utils.image_dataset_from_directory(
        validation_data_path,
        image_size=(image_height, image_width),
        color_mode='grayscale',
        batch_size=batch_size
    )

    return training_dataset, validation_dataset


@tf.autograph.experimental.do_not_convert
def compile_model(
        training_dataset: tf.data.Dataset,
        image_width: int=IMAGE_WIDTH,
        image_height: int=IMAGE_HEIGHT,
        learning_rate: float=LEARNING_RATE,
        l1: float=L1_PENALTY,
        l2: float=L2_PENALTY,
        filter_nums=FILTER_NUMS,
        filter_size=FILTER_SIZE
) -> tf.keras.Model:

    '''Builds the convolutional neural network classification model'''

    # Define a data augmentation mini-model to use as an input layer in
    # the classification model
    data_augmentation=keras.Sequential(
        [
            layers.RandomFlip(
                'horizontal',
                input_shape=(image_height,image_width,1)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    # Set the weight initializer function
    initializer=tf.keras.initializers.GlorotUniform(seed=315)

    # Set-up the L1L2 for the dense layers
    regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)

    # Define the model layers in order
    model=Sequential([
        # layers.Input((image_width, image_height, 1)),
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(
            filter_nums[0],
            filter_size,
            activation='relu',
            kernel_initializer=initializer
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(
            filter_nums[1],
            filter_size,
            activation='relu',
            kernel_initializer=initializer
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(
            filter_nums[2],
            filter_size,
            activation='relu',
            kernel_initializer=initializer
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(
            128,
            kernel_regularizer=regularizer,
            activation='relu',
            kernel_initializer=initializer
        ),
        layers.Dense(1)
    ])

    # Define the optimizer
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the model, specifying the type of loss to use during training and any extra
    # metrics to evaluate
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=optimizer,
        metrics=['binary_accuracy']
    )

    return model


def single_training_run(
        training_data_path: str,
        validation_data_path: str,
        image_width: int=IMAGE_WIDTH,
        aspect_ratio: float=ASPECT_RATIO,
        batch_size: int=BATCH_SIZE,
        learning_rate: float=LEARNING_RATE,
        l1_penalty: float=L1_PENALTY,
        l2_penalty: float=L2_PENALTY,
        filter_nums: list=FILTER_NUMS,
        filter_size: int=FILTER_SIZE,
        epochs: int=SINGLE_TRAINING_RUN_EPOCHS,
        steps_per_epoch: int=STEPS_PER_EPOCH,
        validation_steps: int=VALIDATION_STEPS,
        return_datasets: bool=False
) -> keras.callbacks.History:

    '''Does one training run.'''

    # Get dictionary of all arguments being passed into function
    named_args={**locals()}

    # Make output file name string using values of arguments from function call
    results_file=f'{EXPERIMENT_DATA_PATH}/single_model_run'

    for key, value in named_args.items():
        if key != 'training_data_path' and key != 'validation_data_path':
            if isinstance(value, list):
                results_file+=f"_{'_'.join(map(str, value))}"
            else:
                results_file+=f'_{value}'

    results_file+='.plk'

    # Calculate the image height from the width and target aspect ratio
    image_height=int(image_width / aspect_ratio)

    # Make the streaming datasets
    training_dataset, validation_dataset=make_datasets(
        training_data_path,
        validation_data_path,
        image_width,
        image_height,
        batch_size
    )

    # Check if we have already run this experiment, if not, run it and save the results
    if os.path.isfile(results_file) is False:

        # Make the model
        model=compile_model(
            training_dataset,
            image_width,
            image_height,
            learning_rate,
            l1_penalty,
            l2_penalty,
            filter_nums,
            filter_size
        )

        # Do the training run
        training_result=model.fit(
            training_dataset.repeat(),
            validation_data=validation_dataset.repeat(),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=False
        )

        # Save the results
        with open(results_file, 'wb') as output_file:
            pickle.dump(training_result, output_file, protocol=pickle.HIGHEST_PROTOCOL)

    # If we have already run it, load the result so we can plot it
    elif os.path.isfile(results_file) is True:

        print('Training run already complete, loading results from disk.')

        with open(results_file, 'rb') as output_file:
            training_result=pickle.load(output_file)

    if return_datasets is True:
        return training_result, training_dataset, validation_dataset

    return training_result


def plot_single_training_run(
        training_results: keras.callbacks.History,
        grid: bool=False,
        log_scale: bool=False
) -> plt:

    '''Takes a training results dictionary, plots it.'''

    # Set-up a 1x2 figure for accuracy and binary cross-entropy
    fig, axs=plt.subplots(1,2, figsize=(8,4))

    # Add the main title
    fig.suptitle('CNN training curves', size='large')

    # Plot training and validation accuracy
    axs[0].set_title('Accuracy')
    axs[0].plot(np.array(training_results.history['binary_accuracy']) * 100, label='Training')
    axs[0].plot(np.array(training_results.history['val_binary_accuracy']) * 100, label='Validation')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy (%)')
    axs[0].legend(loc='best')

    if grid is True:
        axs[0].grid(which='both', axis='y')

    if log_scale is True:
        axs[0].set_yscale('log')

    # Plot training and validation binary cross-entropy
    axs[1].set_title('Binary cross-entropy')
    axs[1].plot(training_results.history['loss'])
    axs[1].plot(training_results.history['val_loss'])
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Binary cross-entropy')

    if grid is True:
        axs[1].grid(which='both', axis='y')

    if log_scale is True:
        axs[1].set_yscale('log')

    fig.tight_layout()

    return plt


def hyperparameter_optimization_run(
        training_data_path: str,
        validation_data_path: str,
        image_widths: int=[IMAGE_WIDTH],
        batch_sizes: list=[BATCH_SIZE],
        learning_rates: list=[LEARNING_RATE],
        l1_penalties: list=[L1_PENALTY],
        l2_penalties: list=[L2_PENALTY],
        filter_nums_list: list=[FILTER_NUMS],
        filter_sizes: int=[FILTER_SIZE],
        aspect_ratio: int=ASPECT_RATIO,
        epochs: int=OPTIMIZATION_TRAINING_RUN_EPOCHS,
        steps_per_epoch: int=STEPS_PER_EPOCH,
        validation_steps: int=VALIDATION_STEPS
) -> keras.callbacks.History:

    '''Does hyperparameter optimization run'''

    # Get dictionary of all arguments being passed into function
    named_args = {**locals()}

    # Make output file name string using values of arguments from function call
    results_file=f'{EXPERIMENT_DATA_PATH}/optimization_run'

    for key, value in named_args.items():
        if key != 'training_data_path' and key != 'validation_data_path':
            if isinstance(value, list):
                results_file+=f"_{'_'.join(map(str, value))}"
            else:
                results_file+=f'_{value}'

    results_file+='.plk'

    # Check if we have already run this experiment, if not, run it and save the results
    if os.path.isfile(results_file) is False:

        # Empty collector for individual run results
        hyperparameter_optimization_results=[]

        # Make a list of condition tuples by taking the cartesian product
        # of the hyperparameter setting lists
        conditions=list(
            itertools.product(
                batch_sizes,
                learning_rates,
                l1_penalties,
                l2_penalties,
                image_widths,
                filter_nums_list,
                filter_sizes
            )
        )

        num_conditions=len(batch_sizes)*len(learning_rates)*len(l1_penalties
            )*len(l2_penalties)*len(image_widths)*len(filter_nums_list)*len(filter_sizes)

        # Loop on the conditions
        for i, condition in enumerate(conditions):

            # Unpack the hyperparameter values from the condition tuple
            batch_size, learning_rate, l1, l2, image_width, filter_nums, filter_size=condition
            print(f'Starting training run {i + 1} of {num_conditions}')

            # Calculate the image height from the width and target aspect ratio
            image_height=int(image_width / aspect_ratio)

            # Make the datasets with the batch size for this run
            training_dataset, validation_dataset=make_datasets(
                training_data_path,
                validation_data_path,
                image_width,
                image_height,
                batch_size
            )

            # Compile the model with the learning rate for this run
            model=compile_model(
                training_dataset,
                image_width,
                image_height,
                learning_rate,
                l1,
                l2,
                filter_nums,
                filter_size
            )

            # Do the training run
            hyperparameter_optimization_result=model.fit(
                training_dataset.repeat(),
                validation_data=validation_dataset.repeat(),
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                verbose=False
            )

            # Collect the results
            hyperparameter_optimization_results.append(hyperparameter_optimization_result)

        # Save the result
        with open(results_file, 'wb') as output_file:
            pickle.dump(
                hyperparameter_optimization_results,
                output_file,
                protocol=pickle.HIGHEST_PROTOCOL
            )

    # If we have already run it, load the result so we can plot it
    elif os.path.isfile(results_file) is True:

        print('Optimization run already complete, loading results from disk.')

        with open(results_file, 'rb') as output_file:
            hyperparameter_optimization_results=pickle.load(output_file)

    return hyperparameter_optimization_results


def plot_hyperparameter_optimization_run(
        hyperparameter_optimization_results: dict,
        hyperparameters: dict,
        plot_labels: list,
        accuracy_ylims: list=(None, None),
        entropy_ylims: list=(None, None)
) -> plt:

    '''Takes hyperparameter optimization results and hyperparameters dictionary, plots.'''

    # Dictionary to translate hyperparameter variable names into formatted versions
    # for printing on plot labels
    translation_dict={
        'batch_sizes': 'batch size',
        'learning_rates': 'learning rate',
        'l1_penalties': 'L1 penalty',
        'l2_penalties': 'L2 penalty',
        'image_widths': 'Image width',
        'filter_nums_list': 'Filter counts',
        'filter_sizes': 'Filter size'
    }

    # Set-up a 1x2 figure for accuracy and binary cross-entropy
    fig, axs=plt.subplots(
        len(hyperparameter_optimization_results),
        2,
        figsize=(8,3*len(hyperparameter_optimization_results))
    )

    # Get just the hyperparameters that are being included in the plot
    # labels
    plot_hyperparameters={}
    for plot_label in plot_labels:
        plot_hyperparameters[plot_label]=hyperparameters[plot_label]

    # Build the list of condition tuples
    conditions=list(itertools.product(*list(plot_hyperparameters.values())))

    # Plot the results of each training run
    for i, parameters in enumerate(conditions):

        # Pull the run out of the results dictionary
        training_result=hyperparameter_optimization_results[i]

        # Build the run condition string for the plot title
        condition_string=[]
        for plot_label, value in zip(plot_labels, parameters):
            plot_label=translation_dict[plot_label]
            condition_string.append(f'{plot_label}: {value}')

        condition_string=', '.join(condition_string)

        # Plot training and validation accuracy
        axs[i,0].set_title(condition_string)
        axs[i,0].plot(np.array(
            training_result.history['binary_accuracy']) * 100,
            label='Training'
        )
        axs[i,0].plot(np.array(
            training_result.history['val_binary_accuracy']) * 100,
            label='Validation'
        )
        axs[i,0].set_xlabel('Epoch')
        axs[i,0].set_ylabel('Accuracy (%)')
        axs[i,0].set_ylim(accuracy_ylims)
        axs[i,0].legend(loc='best')

        # Plot training and validation binary cross-entropy
        axs[i,1].set_title(condition_string)
        axs[i,1].plot(training_result.history['loss'], label='Training')
        axs[i,1].plot(training_result.history['val_loss'], label='Validation')
        axs[i,1].set_xlabel('Epoch')
        axs[i,1].set_ylabel('Binary cross-entropy')
        axs[i,1].set_ylim(entropy_ylims)
        axs[i,1].legend(loc='best')

    fig.tight_layout()

    return plt


####################################################################
# Data preparation functions #######################################
####################################################################

def prep_data() -> Tuple[str, str]:

    '''Looks at working directory to determine if we are running 
    in a Kaggle notebook or not. Prepares data accordingly. Returns 
    paths to training and testing datasets.'''
    
    working_directory=os.getcwd()
    path_list=working_directory.split(os.sep)
    path_list=[s for s in path_list if s]

    if len(path_list) >= 1:
        if path_list[0] == 'kaggle':
            environment='kaggle'

        else:
            environment='other'
    else:
        environment='other'

    if environment == 'other':
        training_data_path, validation_data_path, testing_data_path=other_env_data_prep()

    elif environment == 'kaggle':
        training_data_path, validation_data_path, testing_data_path=kaggle_env_data_prep()

    else:
        print('Could not determine environment')
        training_data_path, testing_data_path=None, None

    return training_data_path, validation_data_path, testing_data_path


def other_env_data_prep() -> Tuple[str, str]:

    '''Organizes data that has already been downloaded via
    get_data.sh in a non-kaggle environment. Returns paths 
    to training and testing datasets.'''

    print('Not running in Kaggle notebook')

    image_directory='../data/images'
    raw_image_directory=f'{image_directory}/raw'
    archive_filepath=f'{raw_image_directory}/dogs-vs-cats.zip'

    print('Checking data prep')
    run_data_prep=check_data_prep(image_directory)

    if run_data_prep is False:
        print('Data prep already complete')

    else:
        print('Running data prep')
        print(f'Image archive should be at {archive_filepath}')

        if Path(archive_filepath).is_file() is False:
            print(f'{archive_filepath} does not exist')

        else:
            if Path(f'{raw_image_directory}/train.zip').is_file() is False:
                print(f'Extracting {archive_filepath}')
                with zipfile.ZipFile(archive_filepath, mode='r') as archive:
                    archive.extract('train.zip', f'{raw_image_directory}/')

            else:
                print(f'dogs-vs-cats.zip already extracted')

            if Path(f'{raw_image_directory}/train').is_dir() is False:
                training_archive_filepath=f'{raw_image_directory}/train.zip'

                with zipfile.ZipFile(training_archive_filepath, mode='r') as archive:
                    for file in archive.namelist():
                        if file.endswith('.jpg'):
                            archive.extract(file, raw_image_directory)

            else:
                print(f'train.zip already extracted')
                
        print('Image extraction complete')
        
        print('Making training and testing datasets')
        copy_images(raw_image_directory, image_directory)
        print('Done')
    

    return '../data/images/training', '../data/images/validation', '../data/images/testing'


def kaggle_env_data_prep() -> Tuple[str, str]:

    '''Organizes data from attached data source in kaggle environment.
    Returns paths to training and testing datasets.'''

    print('Running in Kaggle notebook')

    image_directory='/kaggle/working/images'
    raw_image_directory=f'{image_directory}/raw'
    archive_filepath='/kaggle/input/dogs-vs-cats/train.zip'

    print('Checking data prep')
    run_data_prep=check_data_prep(image_directory)

    if run_data_prep is False:
        print('Data prep already complete')

    else:
        print('Running data prep')
        print(f'Image archive should be at {archive_filepath}')

        if Path(archive_filepath).is_file() is False:
            print(f'{archive_filepath} does not exist')

        else:
            if Path(f'{raw_image_directory}/train').is_dir() is False:
                Path(f'{raw_image_directory}/train').mkdir(parents=True)
                with zipfile.ZipFile(archive_filepath, mode='r') as archive:
                    for file in archive.namelist():
                        if file.endswith('.jpg'):
                            archive.extract(file, raw_image_directory)

            else:
                print(f'train.zip already extracted')
                
        print('Image extraction complete')
        print('Making training and testing datasets')
        copy_images(raw_image_directory, image_directory)
        print('Done')

    return f'{image_directory}/training', f'{image_directory}/validation', f'{image_directory}/testing'


def check_data_prep(image_directory: str) -> bool:

    '''Takes string path to image directory. Checks training 
    and testing directories and image counts, returns True 
    or False if data preparation is complete.'''

    run_data_prep=False

    dataset_directories=[
        'training/cats',
        'training/dogs',
        'validation/cats',
        'validation/dogs',
        'testing/cats',
        'testing/dogs',
    ]

    for dataset_directory in dataset_directories:
        if Path(f'{image_directory}/{dataset_directory}').is_dir() is False:
            print(f'Missing {image_directory}/{dataset_directory}')
            run_data_prep=True

    image_count=0

    if run_data_prep is False:
        for dataset_directory in dataset_directories:
            images=glob.glob(f'{image_directory}/{dataset_directory}/*.jpg')
            image_count+=len(images)

        if image_count != 25000:
            print(f'Missing images, final count: {image_count}')
            run_data_prep=True

    return run_data_prep


def copy_images(raw_image_directory: str, image_directory: str) -> None:

    '''Takes string paths to image directory and raw image directory, splits
    cats and dogs into training and testing subsets and copies each to 
    corresponding subdirectory.'''

    # Get a list of dog and cat images
    dogs=glob.glob(f'{raw_image_directory}/train/dog.*')
    cats=glob.glob(f'{raw_image_directory}/train/cat.*')

    # Shuffle
    random.shuffle(dogs)
    random.shuffle(cats)

    num_training_dogs=int(len(dogs) * 0.6)
    num_training_cats=int(len(cats) * 0.6)
    num_validation_dogs=int(len(dogs) * 0.2)
    num_validation_cats=int(len(cats) * 0.2)

    training_dogs=dogs[0:num_training_dogs]
    training_cats=cats[0:num_training_cats]
    validation_cats=cats[num_training_cats:num_training_cats+num_validation_cats]
    validation_dogs=cats[num_training_dogs:num_training_dogs+num_validation_dogs]
    testing_dogs=dogs[num_training_dogs+num_validation_dogs:]
    testing_cats=cats[num_training_cats+num_validation_cats:]

    print('Moving files to training, validation & testing, cat & dog subdirectories')

    datasets={
        'training/cats': training_cats,
        'training/dogs': training_dogs,
        'validation/cats': validation_cats,
        'validation/dogs': validation_dogs,
        'testing/cats': testing_cats,
        'testing/dogs': testing_dogs
    }

    for dataset, image_paths in datasets.items():
        dataset_path=f'{image_directory}/{dataset}'

        Path(dataset_path).mkdir(parents=True, exist_ok=True)
        
        for filepath in image_paths:
            filename=os.path.basename(filepath)
            shutil.copy(
                filepath,
                f'{dataset_path}/{filename}'
            )