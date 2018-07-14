# define important folders
import os
import Modules
import glob
import numpy as np
from KS_lib.prepare_data import routine
from KS_lib.image import KSimage

he_dir = os.path.join('HE')
he_cell_segmentation_result_path = os.path.join('Result')
he_dcis_segmentation_result_path = os.path.join('Result_tumour')

dict_path = {'he': he_dir}
dict_ext = {'he': '.tiff'}
gpu_list = ['0']

#######################################################################
# generate mask
mask_path = 'Mask'
routine.create_dir(mask_path)

files = glob.glob(os.path.join(dict_path['he'], '*' + dict_ext['he']))

for file in files:
    basename = os.path.basename(file)
    basename = os.path.splitext(basename)[0]
    savename = os.path.join(mask_path, basename + '.png')

    I = KSimage.imread(file)
    mask = 255 * np.ones(shape=(I.shape[0], I.shape[1]), dtype=np.uint8)
    KSimage.imwrite(mask, savename)

#######################################################################
# he cell segmentation
Modules.he_cell_segmentation(he_dir,
                             dict_ext,
                             mask_path,
                             he_cell_segmentation_result_path,
                             gpu_list)


# he tumour_seg
Modules.he_dcis_segmentation(he_dir,
                             dict_ext,
                             he_dcis_segmentation_result_path,
                             gpu_list)
