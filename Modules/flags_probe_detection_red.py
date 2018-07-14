"""
Declare all global variables
"""
import os

flags = dict()

# static folder
flags['experiment_folder'] = os.path.join('Modules', 'experiment_probe_detection_red')

flags['size_input_patch'] = [51, 51, 3]
flags['size_output_patch'] = [51, 51, 1]

flags['gpu'] = '/gpu:0'
flags['gpu_memory_fraction'] = 0.7

flags['n_classes'] = 2
flags['alpha'] = 1.0

###################################################################################
# test
flags['test_batch_size'] = 64
flags['stride_test'] = [38, 38]
flags['test_model'] = 0
