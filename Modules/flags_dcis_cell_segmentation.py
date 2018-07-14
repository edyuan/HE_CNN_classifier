"""
Declare all global variables
"""
import os

flags = dict()

# static folder
flags['experiment_folder'] = os.path.join('Modules', 'experiment_dcis_cell_segmentation')

flags['size_input_patch'] = [144, 144, 3]
flags['size_output_patch'] = [144, 144, 1]

flags['gpu'] = '/gpu:0'
flags['gpu_memory_fraction'] = 0.7

flags['n_classes'] = 3
flags['alpha'] = 1.0

###################################################################################
# test
flags['test_batch_size'] = 2
flags['stride_test'] = [96, 96]
flags['test_model'] = 0
