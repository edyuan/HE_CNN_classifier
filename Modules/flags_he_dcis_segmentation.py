"""
Declare all global variables
"""
import os

flags = dict()

# static folder
flags['experiment_folder'] = os.path.join(os.path.join('Modules', 'experiment_he_dcis_segmentation'))

###################################################################################
# training
flags['gpu'] = '/gpu:0'
flags['min_fraction_of_examples_in_queue'] = 0.01
flags['gpu_memory_fraction'] = 0.7

flags['n_classes'] = 2
flags['alpha'] = 1.0

flags['size_input_patch'] = [128, 128, 3]
flags['size_output_patch'] = [128, 128, 2]

###################################################################################
# test
flags['test_batch_size'] = 64
flags['stride_test'] = [40, 40]
flags['test_model'] = 0
