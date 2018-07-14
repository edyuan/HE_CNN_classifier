import os
import numpy as np
import glob
from KS_lib.prepare_data import routine
from Modules.flags_he_cell_segmentation import flags
from KS_lib.tf_model_he_cell_segmentation import utils, tf_model_train
from KS_lib.general import matlab

########################################################################
# define parameters for training the segmentation model

# path to the data structure
flags['dict_path'] = {'HE': os.path.join('Data', 'HE'),
                      'group': os.path.join('Data', 'group')}
# extension of files
flags['dict_ext'] = {'HE': '.png',
                     'group': '.csv'}
# experiment folder  	
flags['experiment_folder'] = os.path.join('Modules', 'experiment_he_cell_segmentation')


flags['num_split'] = 1          # number of folds
flags['split_method'] = 'perm'  # method {'perm', 'val'}
# 'perm': randomly split the data for each of the fold.
#         Thus, there is no guarantee that the same sample will not be used for the same purpose in different folds.
# 'cv': cross-validation

flags['val_percentage'] = 5  # ignored if split_method = 'cv'
flags['test_percentage'] = 10  # ignored if split_method = 'cv'

flags['gen_train_val_method'] = 'sliding_window'  # using sliding window to generate training/validation patches
flags['stride'] = [512, 512]                        # stride for sliding window
flags['HE_thresh'] = 191        # only images with mean intensity less than this are used

flags['HE_ext'] = '.png'         # extension of sliding window patches
flags['DAPI_ext'] = '.png'   # extension of sliding window patches

flags['min_fraction_of_examples_in_queue'] = 0.1  #normally 0.01 # minimum fraction of examples in queue.
# If there are 10,000 patches in the training split, 100 patches will be pre-read into an input queue

flags['num_preprocess_threads'] = 8     # the number of threads to read patches into the queue
flags['batch_size'] = 32                # training/validation batch size

flags['size_input_patch'] = [512, 512, 3]       # size of input
flags['size_output_patch'] = [512, 512, 1]      # size of output
flags['n_classes'] = 4                          # three classes: background, cell, and boundary
flags['class_counts'] = np.zeros(flags['n_classes'])

flags['num_epochs'] = 100                   # number of epochs
flags['initial_learning_rate'] = 1e-4       # learning rate for ADAM
flags['num_examples_per_epoch_for_train'] = 20000   # number of training patches per epoch
flags['num_examples_per_epoch_for_val'] = 5000      # number of validation pathces per epoch
flags['gpu'] = '/gpu:0'                     # fix the gpu
flags['gpu_memory_fraction'] = 0.7          # occupy 70% of the gpu memory

# don't need to care about this
flags['augmentation_keyword'] = 'aug'
flags['alpha'] = 1.0

#######################################################################################
# split data into train, val, test
routine.split_data(flags)     # this will create several csv files in 'path_to_experiment_folder/perm_nth_fold_or_cv_nth_fold'
                              # containing lists of images that will be used fro training, validation, and test


#######################################################################################
# train the model
nth_fold = 1        # indicate which perm/cv fold to train
routine.gen_train_val_data(nth_fold=nth_fold, flags=flags)

# figure which method 'cv' or 'perm' is used
list_dir = os.listdir(os.path.join(flags['experiment_folder']))
if ('cv' + str(nth_fold) in list_dir) and ('perm' + str(nth_fold) in list_dir):
    raise ValueError('Dangerous! You have both cv and perm on the path.')
elif 'cv' + str(nth_fold) in list_dir:
    object_folder = os.path.join(flags['experiment_folder'], 'cv' + str(nth_fold))
elif 'perm' + str(nth_fold) in list_dir:
    object_folder = os.path.join(flags['experiment_folder'], 'perm' + str(nth_fold))
else:
    raise ValueError('No cv or perm folder!')


# create checkpoint folder to save checkpoint files
checkpoint_folder = os.path.join(object_folder, 'checkpoint')

# calculate mean and variance images
network_stats_file_path = os.path.join(checkpoint_folder, 'network_stats.mat')

train_images_folder = os.path.join(object_folder, 'train', 'HE')
if not os.path.isfile(network_stats_file_path):
    utils.calculate_mean_variance_image(object_folder,flags)

# train
tf_model_train.train(object_folder, flags)
