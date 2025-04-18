{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6319d24d",
   "metadata": {
    "_cell_guid": "f9a9d40d-193a-444b-8025-7a749cdc4b82",
    "_uuid": "12732f38-70e5-4c41-a28a-9c0a2093f48d",
    "collapsed": false,
    "execution": {
     "iopub.execute_input": "2025-04-20T00:35:40.568880Z",
     "iopub.status.busy": "2025-04-20T00:35:40.568509Z",
     "iopub.status.idle": "2025-04-20T00:35:56.742489Z",
     "shell.execute_reply": "2025-04-20T00:35:56.741449Z"
    },
    "jupyter": {
     "outputs_hidden": false
    },
    "papermill": {
     "duration": 16.180724,
     "end_time": "2025-04-20T00:35:56.744516",
     "exception": false,
     "start_time": "2025-04-20T00:35:40.563792",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "'''Helper function for training of convolutional neural network classifier.'''\n",
    "\n",
    "# Handle imports up-front\n",
    "import os\n",
    "import itertools\n",
    "import pickle\n",
    "import zipfile\n",
    "import shutil\n",
    "import glob\n",
    "import random\n",
    "from typing import Tuple\n",
    "from pathlib import Path\n",
    "\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers\n",
    "from tensorflow.keras.models import Sequential\n",
    "\n",
    "\n",
    "# Figure out if we are in a Kaggle notebook or not\n",
    "working_directory=os.getcwd()\n",
    "path_list=working_directory.split(os.sep)\n",
    "path_list=[s for s in path_list if s]\n",
    "\n",
    "if len(path_list) >= 1:\n",
    "    if path_list[0] == 'kaggle':\n",
    "        environment='kaggle'\n",
    "\n",
    "    else:\n",
    "        environment='other'\n",
    "else:\n",
    "    environment='other'\n",
    "\n",
    "# Set experiment save file directory accordingly\n",
    "if environment == 'kaggle':\n",
    "    EXPERIMENT_DATA_PATH='/kaggle/working/experiment_results'\n",
    "\n",
    "else:\n",
    "    EXPERIMENT_DATA_PATH='../data/experiment_results'\n",
    "\n",
    "# Make the experiment save file directory\n",
    "Path(EXPERIMENT_DATA_PATH).mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "####################################################################\n",
    "# Convolutional neural network training and optimization functions #\n",
    "####################################################################\n",
    "\n",
    "# Defaults for image parameters\n",
    "IMAGE_HEIGHT=96\n",
    "IMAGE_WIDTH=128\n",
    "ASPECT_RATIO=4/3\n",
    "GRAYSCALE=False\n",
    "\n",
    "# Set some global default values for how long/how much to train\n",
    "BATCH_SIZE=32\n",
    "LEARNING_RATE=0.01\n",
    "SINGLE_TRAINING_RUN_EPOCHS=20\n",
    "OPTIMIZATION_TRAINING_RUN_EPOCHS=10\n",
    "STEPS_PER_EPOCH=10\n",
    "VALIDATION_STEPS=10\n",
    "\n",
    "# Defaults for some CNN parameters\n",
    "L1_PENALTY=None\n",
    "L2_PENALTY=None\n",
    "FILTER_NUMS=[16,32,64]\n",
    "FILTER_SIZE=3\n",
    "\n",
    "\n",
    "def make_datasets(\n",
    "        training_data_path: str,\n",
    "        validation_data_path: str,\n",
    "        image_height: int=IMAGE_HEIGHT,\n",
    "        image_width: int=IMAGE_WIDTH,\n",
    "        grayscale: bool=GRAYSCALE,\n",
    "        batch_size: int=BATCH_SIZE\n",
    ") -> Tuple[tf.data.Dataset, tf.data.Dataset]:\n",
    "\n",
    "    '''Makes training and validation dataset generator objects.'''\n",
    "\n",
    "    if grayscale is True:\n",
    "        color_mode='grayscale'\n",
    "    else:\n",
    "        color_mode='rgb'\n",
    "\n",
    "    training_dataset=tf.keras.utils.image_dataset_from_directory(\n",
    "        training_data_path,\n",
    "        image_size=(image_height, image_width),\n",
    "        color_mode=color_mode,\n",
    "        batch_size=batch_size\n",
    "    )\n",
    "\n",
    "    validation_dataset=tf.keras.utils.image_dataset_from_directory(\n",
    "        validation_data_path,\n",
    "        image_size=(image_height, image_width),\n",
    "        color_mode=color_mode,\n",
    "        batch_size=batch_size\n",
    "    )\n",
    "\n",
    "    return training_dataset, validation_dataset\n",
    "\n",
    "\n",
    "#@tf.autograph.experimental.do_not_convert\n",
    "def compile_model(\n",
    "        image_height: int=IMAGE_HEIGHT,\n",
    "        image_width: int=IMAGE_WIDTH,\n",
    "        grayscale: bool=GRAYSCALE,\n",
    "        learning_rate: float=LEARNING_RATE,\n",
    "        l1: float=L1_PENALTY,\n",
    "        l2: float=L2_PENALTY,\n",
    "        filter_nums=FILTER_NUMS,\n",
    "        filter_size=FILTER_SIZE\n",
    ") -> tf.keras.Model:\n",
    "\n",
    "    '''Builds the convolutional neural network classification model'''\n",
    "\n",
    "    if grayscale is True:\n",
    "        channels=1\n",
    "    else:\n",
    "        channels=3\n",
    "\n",
    "    # Define a data augmentation mini-model to use as an input layer in\n",
    "    # the classification model\n",
    "    data_augmentation=keras.Sequential(\n",
    "        [\n",
    "            layers.Input((image_height,image_width,channels)),\n",
    "            layers.RandomFlip('horizontal'),\n",
    "            layers.RandomRotation(0.1),\n",
    "            layers.RandomZoom(0.1),\n",
    "        ]\n",
    "    )\n",
    "\n",
    "    # Set-up the L1L2 for the dense layers\n",
    "    regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2)\n",
    "\n",
    "    # Define the model layers in order\n",
    "    model=Sequential([\n",
    "        layers.Input((image_height,image_width,channels)),\n",
    "        data_augmentation,\n",
    "        layers.Rescaling(1./255),\n",
    "        layers.Conv2D(\n",
    "            filter_nums[0],\n",
    "            filter_size,\n",
    "            padding='same',\n",
    "            activation='relu',\n",
    "        ),\n",
    "        layers.MaxPooling2D(),\n",
    "        layers.Conv2D(\n",
    "            filter_nums[1],\n",
    "            filter_size,\n",
    "            padding='same',\n",
    "            activation='relu',\n",
    "        ),\n",
    "        layers.MaxPooling2D(),\n",
    "        layers.Conv2D(\n",
    "            filter_nums[2],\n",
    "            filter_size,\n",
    "            padding='same',\n",
    "            activation='relu',\n",
    "        ),\n",
    "        layers.MaxPooling2D(),\n",
    "        # layers.Dropout(0.2),\n",
    "        layers.Flatten(),\n",
    "        layers.Dense(\n",
    "            128,\n",
    "            kernel_regularizer=regularizer,\n",
    "            activation='relu',\n",
    "        ),\n",
    "        layers.Dense(1, activation='sigmoid')\n",
    "    ])\n",
    "\n",
    "    # Define the optimizer\n",
    "    optimizer=keras.optimizers.Adam(learning_rate=learning_rate)\n",
    "\n",
    "    # Compile the model, specifying the type of loss to use during training and any extra\n",
    "    # metrics to evaluate\n",
    "    model.compile(\n",
    "        optimizer=optimizer,\n",
    "        loss='binary_crossentropy',\n",
    "        metrics=['binary_accuracy']\n",
    "    )\n",
    "\n",
    "    return model\n",
    "\n",
    "\n",
    "def single_training_run(\n",
    "        training_data_path: str,\n",
    "        validation_data_path: str,\n",
    "        image_height: int=IMAGE_HEIGHT,\n",
    "        image_width: int=IMAGE_WIDTH,\n",
    "        grayscale: bool=GRAYSCALE,\n",
    "        batch_size: int=BATCH_SIZE,\n",
    "        learning_rate: float=LEARNING_RATE,\n",
    "        l1_penalty: float=L1_PENALTY,\n",
    "        l2_penalty: float=L2_PENALTY,\n",
    "        filter_nums: list=FILTER_NUMS,\n",
    "        filter_size: int=FILTER_SIZE,\n",
    "        epochs: int=SINGLE_TRAINING_RUN_EPOCHS,\n",
    "        steps_per_epoch: int=STEPS_PER_EPOCH,\n",
    "        validation_steps: int=VALIDATION_STEPS,\n",
    "        return_datasets: bool=False,\n",
    "        verbose: bool=False,\n",
    "        print_model_summary: bool=False\n",
    ") -> keras.callbacks.History:\n",
    "\n",
    "    '''Does one training run.'''\n",
    "\n",
    "    # Get dictionary of all arguments being passed into function\n",
    "    named_args={**locals()}\n",
    "\n",
    "    # Make output file name string using values of arguments from function call\n",
    "    results_file=f'{EXPERIMENT_DATA_PATH}/single_model_run'\n",
    "\n",
    "    for key, value in named_args.items():\n",
    "        if key != 'training_data_path' and key != 'validation_data_path':\n",
    "            if isinstance(value, list):\n",
    "                results_file+=f\"_{'_'.join(map(str, value))}\"\n",
    "            else:\n",
    "                results_file+=f'_{value}'\n",
    "\n",
    "    results_file+='.plk'\n",
    "\n",
    "    # Make the streaming datasets\n",
    "    training_dataset, validation_dataset=make_datasets(\n",
    "        training_data_path,\n",
    "        validation_data_path,\n",
    "        image_height,\n",
    "        image_width,\n",
    "        grayscale,\n",
    "        batch_size\n",
    "    )\n",
    "\n",
    "    # If the user set a limited number of steps per epoch, provide infinite datasets\n",
    "    if steps_per_epoch != None:\n",
    "        training_dataset=training_dataset.repeat()\n",
    "\n",
    "    if validation_steps != None:\n",
    "        validation_dataset=validation_dataset.repeat()\n",
    "\n",
    "    # Check if we have already run this experiment, if not, run it and save the results\n",
    "    if os.path.isfile(results_file) is False:\n",
    "\n",
    "        # Make the model\n",
    "        model=compile_model(\n",
    "            image_height,\n",
    "            image_width,\n",
    "            grayscale,\n",
    "            learning_rate,\n",
    "            l1_penalty,\n",
    "            l2_penalty,\n",
    "            filter_nums,\n",
    "            filter_size\n",
    "        )\n",
    "\n",
    "        # Print the model summary, if desired\n",
    "        if print_model_summary is True:\n",
    "            print(f'{model.summary()}\\n')\n",
    "\n",
    "        # Do the training run\n",
    "        training_result=model.fit(\n",
    "            training_dataset,\n",
    "            validation_data=validation_dataset,\n",
    "            epochs=epochs,\n",
    "            steps_per_epoch=steps_per_epoch,\n",
    "            validation_steps=validation_steps,\n",
    "            verbose=verbose\n",
    "        )\n",
    "\n",
    "        # Save the results\n",
    "        with open(results_file, 'wb') as output_file:\n",
    "            pickle.dump(training_result, output_file, protocol=pickle.HIGHEST_PROTOCOL)\n",
    "\n",
    "    # If we have already run it, load the result so we can plot it\n",
    "    elif os.path.isfile(results_file) is True:\n",
    "\n",
    "        print('Training run already complete, loading results from disk.')\n",
    "\n",
    "        # Load the results object from disk\n",
    "        with open(results_file, 'rb') as output_file:\n",
    "            training_result=pickle.load(output_file)\n",
    "\n",
    "    # Return the datasets and training results if called for\n",
    "    if return_datasets is True:\n",
    "        return training_result, training_dataset, validation_dataset\n",
    "\n",
    "    # Or, return just the training result\n",
    "    return training_result\n",
    "\n",
    "\n",
    "def plot_single_training_run(\n",
    "        training_results: keras.callbacks.History,\n",
    "        grid: bool=False,\n",
    "        log_scale: bool=False\n",
    ") -> plt:\n",
    "\n",
    "    '''Takes a training results dictionary, plots it.'''\n",
    "\n",
    "    # Set-up a 1x2 figure for accuracy and binary cross-entropy\n",
    "    fig, axs=plt.subplots(1,2, figsize=(8,4))\n",
    "\n",
    "    # Add the main title\n",
    "    fig.suptitle('CNN training curves', size='large')\n",
    "\n",
    "    # Plot training and validation accuracy\n",
    "    axs[0].set_title('Accuracy')\n",
    "    axs[0].plot(np.array(training_results.history['binary_accuracy']) * 100, label='Training')\n",
    "    axs[0].plot(np.array(training_results.history['val_binary_accuracy']) * 100, label='Validation')\n",
    "    axs[0].set_xlabel('Epoch')\n",
    "    axs[0].set_ylabel('Accuracy (%)')\n",
    "    axs[0].legend(loc='best')\n",
    "\n",
    "    if grid is True:\n",
    "        axs[0].grid(which='both', axis='y')\n",
    "\n",
    "    if log_scale is True:\n",
    "        axs[0].set_yscale('log')\n",
    "\n",
    "    # Plot training and validation binary cross-entropy\n",
    "    axs[1].set_title('Binary cross-entropy')\n",
    "    axs[1].plot(training_results.history['loss'])\n",
    "    axs[1].plot(training_results.history['val_loss'])\n",
    "    axs[1].set_xlabel('Epoch')\n",
    "    axs[1].set_ylabel('Binary cross-entropy')\n",
    "\n",
    "    if grid is True:\n",
    "        axs[1].grid(which='both', axis='y')\n",
    "\n",
    "    if log_scale is True:\n",
    "        axs[1].set_yscale('log')\n",
    "\n",
    "    fig.tight_layout()\n",
    "\n",
    "    return plt\n",
    "\n",
    "\n",
    "def hyperparameter_optimization_run(\n",
    "        training_data_path: str,\n",
    "        validation_data_path: str,\n",
    "        image_widths: int=[IMAGE_WIDTH],\n",
    "        batch_sizes: list=[BATCH_SIZE],\n",
    "        learning_rates: list=[LEARNING_RATE],\n",
    "        l1_penalties: list=[L1_PENALTY],\n",
    "        l2_penalties: list=[L2_PENALTY],\n",
    "        filter_nums_list: list=[FILTER_NUMS],\n",
    "        filter_sizes: int=[FILTER_SIZE],\n",
    "        aspect_ratio: int=ASPECT_RATIO,\n",
    "        grayscale: bool=GRAYSCALE,\n",
    "        epochs: int=OPTIMIZATION_TRAINING_RUN_EPOCHS,\n",
    "        steps_per_epoch: int=STEPS_PER_EPOCH,\n",
    "        validation_steps: int=VALIDATION_STEPS\n",
    ") -> keras.callbacks.History:\n",
    "\n",
    "    '''Does hyperparameter optimization run'''\n",
    "\n",
    "    # Get dictionary of all arguments being passed into function\n",
    "    named_args = {**locals()}\n",
    "\n",
    "    # Make output file name string using values of arguments from function call\n",
    "    results_file=f'{EXPERIMENT_DATA_PATH}/optimization_run'\n",
    "\n",
    "    for key, value in named_args.items():\n",
    "        if key != 'training_data_path' and key != 'validation_data_path':\n",
    "            if isinstance(value, list):\n",
    "                if key == 'filter_nums_list':\n",
    "                    results_file+=f\"_{'_'.join(map(str, itertools.chain(*value)))}\"\n",
    "                else:\n",
    "                    results_file+=f\"_{'_'.join(map(str, value))}\"\n",
    "\n",
    "            else:\n",
    "                results_file+=f'_{value}'\n",
    "\n",
    "    results_file+='.plk'\n",
    "\n",
    "    # Check if we have already run this experiment, if not, run it and save the results\n",
    "    if os.path.isfile(results_file) is False:\n",
    "\n",
    "        # Empty collector for individual run results\n",
    "        hyperparameter_optimization_results=[]\n",
    "\n",
    "        # Make a list of condition tuples by taking the cartesian product\n",
    "        # of the hyperparameter setting lists\n",
    "        conditions=list(\n",
    "            itertools.product(\n",
    "                batch_sizes,\n",
    "                learning_rates,\n",
    "                l1_penalties,\n",
    "                l2_penalties,\n",
    "                image_widths,\n",
    "                filter_nums_list,\n",
    "                filter_sizes\n",
    "            )\n",
    "        )\n",
    "\n",
    "        num_conditions=len(batch_sizes)*len(learning_rates)*len(l1_penalties\n",
    "            )*len(l2_penalties)*len(image_widths)*len(filter_nums_list)*len(filter_sizes)\n",
    "\n",
    "        # Loop on the conditions\n",
    "        for i, condition in enumerate(conditions):\n",
    "\n",
    "            # Unpack the hyperparameter values from the condition tuple\n",
    "            batch_size, learning_rate, l1, l2, image_width, filter_nums, filter_size=condition\n",
    "            print(f'Starting training run {i + 1} of {num_conditions}')\n",
    "\n",
    "            # Calculate the image height from the width and target aspect ratio\n",
    "            image_height=int(image_width / aspect_ratio)\n",
    "\n",
    "            # Make the datasets with the batch size for this run\n",
    "            training_dataset, validation_dataset=make_datasets(\n",
    "                training_data_path,\n",
    "                validation_data_path,\n",
    "                image_height,\n",
    "                image_width,\n",
    "                grayscale,\n",
    "                batch_size\n",
    "            )\n",
    "\n",
    "            # If the user set a limited number of steps per epoch, provide infinite datasets\n",
    "            if steps_per_epoch != None:\n",
    "                training_dataset=training_dataset.repeat()\n",
    "\n",
    "            if validation_steps != None:\n",
    "                validation_dataset=validation_dataset.repeat()\n",
    "\n",
    "            # Compile the model with the learning rate for this run\n",
    "            model=compile_model(\n",
    "                image_height,\n",
    "                image_width,\n",
    "                grayscale,\n",
    "                learning_rate,\n",
    "                l1,\n",
    "                l2,\n",
    "                filter_nums,\n",
    "                filter_size\n",
    "            )\n",
    "\n",
    "            # Do the training run\n",
    "            hyperparameter_optimization_result=model.fit(\n",
    "                training_dataset,\n",
    "                validation_data=validation_dataset,\n",
    "                epochs=epochs,\n",
    "                steps_per_epoch=steps_per_epoch,\n",
    "                validation_steps=validation_steps,\n",
    "                verbose=False\n",
    "            )\n",
    "\n",
    "            # Collect the results\n",
    "            hyperparameter_optimization_results.append(hyperparameter_optimization_result)\n",
    "\n",
    "        # Save the result\n",
    "        with open(results_file, 'wb') as output_file:\n",
    "            pickle.dump(\n",
    "                hyperparameter_optimization_results,\n",
    "                output_file,\n",
    "                protocol=pickle.HIGHEST_PROTOCOL\n",
    "            )\n",
    "\n",
    "    # If we have already run it, load the result so we can plot it\n",
    "    elif os.path.isfile(results_file) is True:\n",
    "\n",
    "        print('Optimization run already complete, loading results from disk.')\n",
    "\n",
    "        with open(results_file, 'rb') as output_file:\n",
    "            hyperparameter_optimization_results=pickle.load(output_file)\n",
    "\n",
    "    return hyperparameter_optimization_results\n",
    "\n",
    "\n",
    "def plot_hyperparameter_optimization_run(\n",
    "        hyperparameter_optimization_results: dict,\n",
    "        hyperparameters: dict,\n",
    "        plot_labels: list,\n",
    "        accuracy_ylims: list=(None, None),\n",
    "        entropy_ylims: list=(None, None)\n",
    ") -> plt:\n",
    "\n",
    "    '''Takes hyperparameter optimization results and hyperparameters dictionary, plots.'''\n",
    "\n",
    "    # Dictionary to translate hyperparameter variable names into formatted versions\n",
    "    # for printing on plot labels\n",
    "    translation_dict={\n",
    "        'batch_sizes': 'batch size',\n",
    "        'learning_rates': 'learning rate',\n",
    "        'l1_penalties': 'L1 penalty',\n",
    "        'l2_penalties': 'L2 penalty',\n",
    "        'image_widths': 'Image width',\n",
    "        'filter_nums_list': 'Filter counts',\n",
    "        'filter_sizes': 'Filter size'\n",
    "    }\n",
    "\n",
    "    # Set-up a 1x2 figure for accuracy and binary cross-entropy\n",
    "    fig, axs=plt.subplots(\n",
    "        len(hyperparameter_optimization_results),\n",
    "        2,\n",
    "        figsize=(8,3*len(hyperparameter_optimization_results))\n",
    "    )\n",
    "\n",
    "    # Get just the hyperparameters that are being included in the plot\n",
    "    # labels\n",
    "    plot_hyperparameters={}\n",
    "    for plot_label in plot_labels:\n",
    "        plot_hyperparameters[plot_label]=hyperparameters[plot_label]\n",
    "\n",
    "    # Build the list of condition tuples\n",
    "    conditions=list(itertools.product(*list(plot_hyperparameters.values())))\n",
    "\n",
    "    # Plot the results of each training run\n",
    "    for i, parameters in enumerate(conditions):\n",
    "\n",
    "        # Pull the run out of the results dictionary\n",
    "        training_result=hyperparameter_optimization_results[i]\n",
    "\n",
    "        # Build the run condition string for the plot title\n",
    "        condition_string=[]\n",
    "        for plot_label, value in zip(plot_labels, parameters):\n",
    "            plot_label=translation_dict[plot_label]\n",
    "            condition_string.append(f'{plot_label}: {value}')\n",
    "\n",
    "        condition_string=', '.join(condition_string)\n",
    "\n",
    "        # Plot training and validation accuracy\n",
    "        axs[i,0].set_title(condition_string)\n",
    "        axs[i,0].plot(np.array(\n",
    "            training_result.history['binary_accuracy']) * 100,\n",
    "            label='Training'\n",
    "        )\n",
    "        axs[i,0].plot(np.array(\n",
    "            training_result.history['val_binary_accuracy']) * 100,\n",
    "            label='Validation'\n",
    "        )\n",
    "        axs[i,0].set_xlabel('Epoch')\n",
    "        axs[i,0].set_ylabel('Accuracy (%)')\n",
    "        axs[i,0].set_ylim(accuracy_ylims)\n",
    "        axs[i,0].legend(loc='best')\n",
    "\n",
    "        # Plot training and validation binary cross-entropy\n",
    "        axs[i,1].set_title(condition_string)\n",
    "        axs[i,1].plot(training_result.history['loss'], label='Training')\n",
    "        axs[i,1].plot(training_result.history['val_loss'], label='Validation')\n",
    "        axs[i,1].set_xlabel('Epoch')\n",
    "        axs[i,1].set_ylabel('Binary cross-entropy')\n",
    "        axs[i,1].set_ylim(entropy_ylims)\n",
    "        axs[i,1].legend(loc='best')\n",
    "\n",
    "    fig.tight_layout()\n",
    "\n",
    "    return plt\n",
    "\n",
    "\n",
    "####################################################################\n",
    "# Data preparation functions #######################################\n",
    "####################################################################\n",
    "\n",
    "def prep_data() -> Tuple[str, str]:\n",
    "\n",
    "    '''Looks at working directory to determine if we are running \n",
    "    in a Kaggle notebook or not. Prepares data accordingly. Returns \n",
    "    paths to training and testing datasets.'''\n",
    "    \n",
    "    working_directory=os.getcwd()\n",
    "    path_list=working_directory.split(os.sep)\n",
    "    path_list=[s for s in path_list if s]\n",
    "\n",
    "    if len(path_list) >= 1:\n",
    "        if path_list[0] == 'kaggle':\n",
    "            environment='kaggle'\n",
    "\n",
    "        else:\n",
    "            environment='other'\n",
    "    else:\n",
    "        environment='other'\n",
    "\n",
    "    if environment == 'other':\n",
    "        training_data_path, validation_data_path, testing_data_path=other_env_data_prep()\n",
    "\n",
    "    elif environment == 'kaggle':\n",
    "        training_data_path, validation_data_path, testing_data_path=kaggle_env_data_prep()\n",
    "\n",
    "    else:\n",
    "        print('Could not determine environment')\n",
    "        training_data_path, testing_data_path=None, None\n",
    "\n",
    "    return training_data_path, validation_data_path, testing_data_path\n",
    "\n",
    "\n",
    "def other_env_data_prep() -> Tuple[str, str]:\n",
    "\n",
    "    '''Organizes data that has already been downloaded via\n",
    "    get_data.sh in a non-kaggle environment. Returns paths \n",
    "    to training and testing datasets.'''\n",
    "\n",
    "    print('Not running in Kaggle notebook')\n",
    "\n",
    "    image_directory='../data/images'\n",
    "    raw_image_directory=f'{image_directory}/raw'\n",
    "    archive_filepath=f'{raw_image_directory}/dogs-vs-cats.zip'\n",
    "\n",
    "    print('Checking data prep')\n",
    "    run_data_prep=check_data_prep(image_directory)\n",
    "\n",
    "    if run_data_prep is False:\n",
    "        print('Data prep already complete')\n",
    "\n",
    "    else:\n",
    "        print('Running data prep')\n",
    "        print(f'Image archive should be at {archive_filepath}')\n",
    "\n",
    "        if Path(archive_filepath).is_file() is False:\n",
    "            print(f'{archive_filepath} does not exist')\n",
    "\n",
    "        else:\n",
    "            if Path(f'{raw_image_directory}/train.zip').is_file() is False:\n",
    "                print(f'Extracting {archive_filepath}')\n",
    "                with zipfile.ZipFile(archive_filepath, mode='r') as archive:\n",
    "                    archive.extract('train.zip', f'{raw_image_directory}/')\n",
    "\n",
    "            else:\n",
    "                print(f'dogs-vs-cats.zip already extracted')\n",
    "\n",
    "            if Path(f'{raw_image_directory}/train').is_dir() is False:\n",
    "                training_archive_filepath=f'{raw_image_directory}/train.zip'\n",
    "\n",
    "                with zipfile.ZipFile(training_archive_filepath, mode='r') as archive:\n",
    "                    for file in archive.namelist():\n",
    "                        if file.endswith('.jpg'):\n",
    "                            archive.extract(file, raw_image_directory)\n",
    "\n",
    "            else:\n",
    "                print(f'train.zip already extracted')\n",
    "                \n",
    "        print('Image extraction complete')\n",
    "        \n",
    "        print('Making training and testing datasets')\n",
    "        copy_images(raw_image_directory, image_directory)\n",
    "        print('Done')\n",
    "    \n",
    "\n",
    "    return '../data/images/training', '../data/images/validation', '../data/images/testing'\n",
    "\n",
    "\n",
    "def kaggle_env_data_prep() -> Tuple[str, str]:\n",
    "\n",
    "    '''Organizes data from attached data source in kaggle environment.\n",
    "    Returns paths to training and testing datasets.'''\n",
    "\n",
    "    print('Running in Kaggle notebook')\n",
    "\n",
    "    image_directory='/kaggle/working/images'\n",
    "    raw_image_directory=f'{image_directory}/raw'\n",
    "    archive_filepath='/kaggle/input/dogs-vs-cats/train.zip'\n",
    "\n",
    "    print('Checking data prep')\n",
    "    run_data_prep=check_data_prep(image_directory)\n",
    "\n",
    "    if run_data_prep is False:\n",
    "        print('Data prep already complete')\n",
    "\n",
    "    else:\n",
    "        print('Running data prep')\n",
    "        print(f'Image archive should be at {archive_filepath}')\n",
    "\n",
    "        if Path(archive_filepath).is_file() is False:\n",
    "            print(f'{archive_filepath} does not exist')\n",
    "\n",
    "        else:\n",
    "            if Path(f'{raw_image_directory}/train').is_dir() is False:\n",
    "                Path(f'{raw_image_directory}/train').mkdir(parents=True)\n",
    "                with zipfile.ZipFile(archive_filepath, mode='r') as archive:\n",
    "                    for file in archive.namelist():\n",
    "                        if file.endswith('.jpg'):\n",
    "                            archive.extract(file, raw_image_directory)\n",
    "\n",
    "            else:\n",
    "                print(f'train.zip already extracted')\n",
    "                \n",
    "        print('Image extraction complete')\n",
    "        print('Making training and testing datasets')\n",
    "        copy_images(raw_image_directory, image_directory)\n",
    "        print('Done')\n",
    "\n",
    "    return f'{image_directory}/training', f'{image_directory}/validation', f'{image_directory}/testing'\n",
    "\n",
    "\n",
    "def check_data_prep(image_directory: str) -> bool:\n",
    "\n",
    "    '''Takes string path to image directory. Checks training \n",
    "    and testing directories and image counts, returns True \n",
    "    or False if data preparation is complete.'''\n",
    "\n",
    "    run_data_prep=False\n",
    "\n",
    "    dataset_directories=[\n",
    "        'training/cats',\n",
    "        'training/dogs',\n",
    "        'validation/cats',\n",
    "        'validation/dogs',\n",
    "        'testing/cats',\n",
    "        'testing/dogs',\n",
    "    ]\n",
    "\n",
    "    for dataset_directory in dataset_directories:\n",
    "        if Path(f'{image_directory}/{dataset_directory}').is_dir() is False:\n",
    "            print(f'Missing {image_directory}/{dataset_directory}')\n",
    "            run_data_prep=True\n",
    "\n",
    "    image_count=0\n",
    "\n",
    "    if run_data_prep is False:\n",
    "        for dataset_directory in dataset_directories:\n",
    "            images=glob.glob(f'{image_directory}/{dataset_directory}/*.jpg')\n",
    "            image_count+=len(images)\n",
    "\n",
    "        if image_count != 25000:\n",
    "            print(f'Missing images, final count: {image_count}')\n",
    "            run_data_prep=True\n",
    "\n",
    "    return run_data_prep\n",
    "\n",
    "\n",
    "def copy_images(raw_image_directory: str, image_directory: str) -> None:\n",
    "\n",
    "    '''Takes string paths to image directory and raw image directory, splits\n",
    "    cats and dogs into training and testing subsets and copies each to \n",
    "    corresponding subdirectory.'''\n",
    "\n",
    "    # Get a list of dog and cat images\n",
    "    dogs=glob.glob(f'{raw_image_directory}/train/dog.*')\n",
    "    cats=glob.glob(f'{raw_image_directory}/train/cat.*')\n",
    "\n",
    "    # Shuffle\n",
    "    random.shuffle(dogs)\n",
    "    random.shuffle(cats)\n",
    "\n",
    "    num_training_dogs=int(len(dogs) * 0.6)\n",
    "    num_training_cats=int(len(cats) * 0.6)\n",
    "    num_validation_dogs=int(len(dogs) * 0.2)\n",
    "    num_validation_cats=int(len(cats) * 0.2)\n",
    "\n",
    "    training_dogs=dogs[0:num_training_dogs]\n",
    "    training_cats=cats[0:num_training_cats]\n",
    "    validation_cats=cats[num_training_cats:num_training_cats+num_validation_cats]\n",
    "    validation_dogs=dogs[num_training_dogs:num_training_dogs+num_validation_dogs]\n",
    "    testing_dogs=dogs[num_training_dogs+num_validation_dogs:]\n",
    "    testing_cats=cats[num_training_cats+num_validation_cats:]\n",
    "\n",
    "    print('Moving files to training, validation & testing, cat & dog subdirectories')\n",
    "\n",
    "    datasets={\n",
    "        'training/cats': training_cats,\n",
    "        'training/dogs': training_dogs,\n",
    "        'validation/cats': validation_cats,\n",
    "        'validation/dogs': validation_dogs,\n",
    "        'testing/cats': testing_cats,\n",
    "        'testing/dogs': testing_dogs\n",
    "    }\n",
    "\n",
    "    for dataset, image_paths in datasets.items():\n",
    "        dataset_path=f'{image_directory}/{dataset}'\n",
    "\n",
    "        Path(dataset_path).mkdir(parents=True, exist_ok=True)\n",
    "        \n",
    "        for filepath in image_paths:\n",
    "            filename=os.path.basename(filepath)\n",
    "            shutil.copy(\n",
    "                filepath,\n",
    "                f'{dataset_path}/{filename}'\n",
    "            )"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [],
   "dockerImageVersionId": 30918,
   "isGpuEnabled": false,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 21.44425,
   "end_time": "2025-04-20T00:35:59.134111",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2025-04-20T00:35:37.689861",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
