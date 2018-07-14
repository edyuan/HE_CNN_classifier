# Segmentation_main
import glob
import os

from KS_lib.tf_model_he_dcis_segmentation import tf_model_test
import numpy as np

def main(nth_fold,mode,flags,test_images_list, result_path, igpu):

    # check if cv or perm
    list_dir = os.listdir(os.path.join(flags['experiment_folder']))
    if ('cv' + str(nth_fold) in list_dir) and ('perm' + str(nth_fold) in list_dir):
        raise ValueError('Dangerous! You have both cv and perm on the path.')
    elif 'cv' + str(nth_fold) in list_dir:
        object_folder = os.path.join(flags['experiment_folder'], 'cv' + str(nth_fold))
    elif 'perm' + str(nth_fold) in list_dir:
        object_folder = os.path.join(flags['experiment_folder'], 'perm' + str(nth_fold))
    else:
        raise ValueError('No cv or perm folder!')

    # Test mode with model specified
    if mode == 'test_model':
        checkpointlist = glob.glob(os.path.join(object_folder, 'checkpoint', 'model*meta'))
        checkpointlist = [file for file in checkpointlist if 'pretrain' not in file]
        temp = []
        for filepath in checkpointlist:
            basename = os.path.basename(filepath)
            temp.append(int(float(basename.split('-')[-1].split('.')[0])))
        temp = np.sort(temp)

        model_path = os.path.join(object_folder, 'checkpoint', 'model.ckpt-' + str(temp[flags['test_model']]))
        print('use epoch %d : model %s' % (flags['test_model'], 'model.ckpt-' + str(temp[flags['test_model']])))
        filename_list = test_images_list
        tf_model_test.test(object_folder, model_path, filename_list, result_path, flags, igpu)
