'''Helper function for training of convolutional neural network classifier.'''


# Handle imports up-front
import os
import itertools
import pickle
from typing import Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

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

SINGLE_TRAINING_RUN_EPOCHS=200
OPTIMIZATION_TRAINING_RUN_EPOCHS=20
STEPS_PER_EPOCH=10
VALIDATION_STEPS=10

# Define a re-usable helper function to create training and validation datasets
def make_datasets(
        training_data_path: str,
        image_width: int=IMAGE_WIDTH,
        image_height: int=IMAGE_HEIGHT,
        batch_size: int=BATCH_SIZE
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    '''Makes training and validation dataset generator objects.'''

    training_dataset, validation_dataset=tf.keras.utils.image_dataset_from_directory(
        training_data_path,
        validation_split=0.2,
        subset='both',
        seed=315,
        shuffle=True,
        image_size=(image_width, image_height),
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

    # Define and adapt a normalization layer. This step uses a sample of images
    # from the training dataset to calculate the mean and standard deviation
    # which will then be used to normalize the data during training
    sample_data=training_dataset.take(1000)
    adapt_data=sample_data.map(lambda x, y: x)
    norm_layer=tf.keras.layers.Normalization()
    norm_layer.adapt(adapt_data)

    # Set the weight initializer function
    initializer=tf.keras.initializers.GlorotUniform(seed=315)

    # Set-up the L1L2 for the dense layers
    regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)

    # Define the model layers in order
    model=Sequential([
        layers.Input((image_width, image_height, 1)),
        norm_layer,
        layers.Conv2D(filter_nums[0], filter_size, activation='relu', kernel_initializer=initializer),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filter_nums[1], filter_size, activation='relu', kernel_initializer=initializer),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filter_nums[2], filter_size, activation='relu', kernel_initializer=initializer),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(
            128,
            kernel_regularizer=regularizer,
            activation='relu',
            kernel_initializer=initializer
        ),
        layers.Dense(
            64,
            kernel_regularizer=regularizer,
            activation='relu',
            kernel_initializer=initializer
        ),
        layers.Dense(1, activation='sigmoid', kernel_initializer=initializer)
    ])

    # Define the optimizer
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the model, specifying the type of loss to use during training and any extra
    # metrics to evaluate
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])

    print(model.summary())

    return model


def single_training_run(
        training_data_path: str,
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
    results_file='../data/experiment_results/single_model_run'

    for key, value in named_args.items():
        if key != 'training_data_path':
            results_file+=f'_{value}'

    results_file+='.plk'

    # Calculate the image height from the width and target aspect ratio
    image_height=int(image_width / aspect_ratio)

    # Make the streaming datasets
    training_dataset, validation_dataset=make_datasets(
        training_data_path,
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
    results_file='../data/experiment_results/optimization_run'

    for key, value in named_args.items():
        if key != 'training_data_path':
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
