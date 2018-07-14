"""
Declare all global variables
"""
import os

flags = dict()

# static folder
flags['experiment_folder'] = os.path.join('Modules', 'experiment_he_cell_segmentation')

###################################################################################
# training

flags['gpu'] = '/gpu:0'
flags['gpu_memory_fraction'] = 0.7

flags['n_classes'] = 3
flags['alpha'] = 1.0

flags['size_input_patch'] = [144, 144, 3]
flags['size_output_patch'] = [144, 144, 1]

###################################################################################
# test
flags['test_batch_size'] = 2
flags['stride_test'] = [96, 96]
flags['test_model'] = 0
