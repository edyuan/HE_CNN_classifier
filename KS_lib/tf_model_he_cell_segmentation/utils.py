from KS_lib.image import KSimage
from KS_lib.prepare_data import routine	
from KS_lib.general import matlab
import numpy as np
import os
import glob

##############################################################################
def calculate_mean_variance_image(object_folder, flags):
    key_values = list(flags['dict_path'].keys())
    key_values.remove('group')
   
    # Setup
    network_stats_file_path = os.path.join(object_folder,'checkpoint','network_stats.mat') 
    routine.create_dir(os.path.join(object_folder,'checkpoint'))
    mean_dict ={}   

 
    for key in key_values:
        image_folder = os.path.join(object_folder, 'train' , key)
        list_images = glob.glob(os.path.join(image_folder, '*' + flags['dict_ext'][key]))

        image = KSimage.imread(list_images[0])
        #if np.random.randint(2, size=1) == 1:
        #    image = np.flipud(image)
        #if np.random.randint(2, size=1) == 1:
        #    image = np.fliplr(image)
        image = np.float32(image)

        mean_image = image
        variance_image = np.zeros(shape=image.shape, dtype=np.float32)

        for t, image_file in enumerate(list_images[1:]):
            image = KSimage.imread(image_file)

            # image = np.dstack((image[:, :, 2], image[:, :, 1], image[:, :, 0]))

            #if np.random.randint(2, size=1) == 1:
            #    image = np.flipud(image)
            #if np.random.randint(2, size=1) == 1:
            #    image = np.fliplr(image)
            image = np.float32(image)

            mean_dict[key +'_mean'] = (np.float32(t + 1) * mean_image + image) / np.float32(t + 2)

            mean_dict[key + '_var'] = np.float32(t + 1) / np.float32(t + 2) * variance_image \
                             + np.float32(1) / np.float32(t + 1) * ((image - mean_image) ** 2)

            print('calculate mean and variance: processing %d out of %d' % (t + 2, len(list_images)))
           
    
    matlab.save(network_stats_file_path, mean_dict)
