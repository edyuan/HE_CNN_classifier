"""
routine.py generates the experiment folder where all experiments will be conducted
"""
import os
import glob
import re
import collections
import time
import csv
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from KS_lib.prepare_data import extract_patches
from KS_lib.general import KScsv
from KS_lib.image import KSimage
from KS_lib.prepare_data import select_instances
from KS_lib.general import matlab
from scipy.ndimage.morphology import binary_erosion
from skimage.morphology import watershed, disk, remove_small_objects, convex_hull_image, dilation
from  scipy import ndimage
from skimage.filters import rank, threshold_otsu
from skimage.measure import regionprops, label
from skimage.feature import register_translation
from skimage.transform import SimilarityTransform
from skimage.transform import warp

################################################################################################
class RegexDict(dict):
    def get_matching(self, event):
        return (self[key] for key in self if re.match(key, event))

    def get_all_matching(self, events):
        return (match for event in events for match in self.get_matching(event))


################################################################################################
def create_dir(dir_name):
    """
    create a directory if not exist.
    :param dir_name:
    :return:
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


################################################################################################
def get_pair_list(dict_path, dict_ext):
    images_list = glob.glob(os.path.join(dict_path['HE'], '*' + dict_ext['HE']))

    #print(dict_path)    
    # get rid of group from dictionary for now
    dict_path_ = dict_path.copy()
    del dict_path_['group']
 
    obj_list = collections.defaultdict(list)

    for image_name in images_list:
        basename = os.path.basename(image_name)
        basename = os.path.splitext(basename)[0]

        dict_name = dict()
        for key in dict_path_.keys():
            dict_name[key] = os.path.join(dict_path[key], basename + dict_ext[key])

        if all(os.path.isfile(v) for k, v in dict_name.items()):
            for key in dict_path_.keys():
                obj_list[key].append(dict_name[key])

    for key in obj_list.keys():
        if not obj_list[key]:
            print("no data in %s" % (dict_path_[key]))
            raise ValueError('terminate!')

    return obj_list


################################################################################################
def split_cv(obj_list, flags):
    """
    split_cv split data into train, validation, and test stratified by the group label
    :param images_list:
    :param labels_list:
    :param groups_list:
    :param num:
    :param val_percentage:
    :return: void but write csv files
    """

    num = flags['num_split']
    val_percentage = flags['val_percentage']

    groups_label = list()
    for file in obj_list['group']:
        row = KScsv.read_csv(file)
        groups_label.append(row[0][0])
    groups_label = np.array(groups_label)

    for key in obj_list.keys():
        obj_list[key] = np.array(obj_list[key])

    skf = StratifiedKFold(n_splits=num)
    for i_num, (train_idx, test_idx) in enumerate(skf.split(obj_list['image'], groups_label)):
        cv_folder = os.path.join(flags['experiment_folder'], 'cv' + str(i_num + 1))
        create_dir(cv_folder)

        test_obj_list_dict = dict()
        train_obj_list_dict = dict()
        for key in obj_list.keys():
            test_obj_list_dict[key] = obj_list[key][test_idx]
            train_obj_list_dict[key] = obj_list[key][train_idx]

        train_groups_label = groups_label[train_idx]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_percentage / 100.0)
        for train_train_index, train_val_index in sss.split(train_obj_list_dict['image'], train_groups_label):
            train_train_obj_list_dict = dict()
            train_val_obj_list_dict = dict()
            for key in train_obj_list_dict.keys():
                train_train_obj_list_dict[key] = train_obj_list_dict[key][train_train_index]
                train_val_obj_list_dict[key] = train_obj_list_dict[key][train_val_index]

        #################################################################
        # test
        for key in test_obj_list_dict.keys():
            filename = os.path.join(cv_folder, 'test_' + key + '_list.csv')
            if not os.path.isfile(filename):
                row_list = [[item] for item in test_obj_list_dict[key]]
                KScsv.write_csv(row_list, filename)

        #################################################################
        # train
        for key in train_train_obj_list_dict.keys():
            filename = os.path.join(cv_folder, 'train_' + key + '_list.csv')
            if not os.path.isfile(filename):
                row_list = [[item] for item in train_train_obj_list_dict[key]]
                KScsv.write_csv(row_list, filename)

        #################################################################
        # validation
        for key in train_val_obj_list_dict.keys():
            filename = os.path.join(cv_folder, 'val_' + key + '_list.csv')
            if not os.path.isfile(filename):
                row_list = [[item] for item in train_val_obj_list_dict[key]]
                KScsv.write_csv(row_list, filename)

################################################################################################
def split_perm(obj_list, flags):
    """
    split_perm split data using permutation with stratification based on group label
    :param images_list:
    :param labels_list:
    :param groups_list:
    :param num:
    :param test_percentage:
    :param val_percentage:
    :return: void
    """
	
    # obj_list - keys - 'HE', 'DAPI', 'weight', 'group' - list
    num = flags['num_split']
    test_percentage = flags['test_percentage']
    val_percentage = flags['val_percentage']

    #temporarily disable group folder
    #groups_label = list()
    #for file in obj_list['group']:
    #    row = KScsv.read_csv(file)
    #    groups_label.append(row[0][0])
    #groups_label = np.array(groups_label)
    groups_label = np.array(['1'] * len(obj_list['HE']))

    for key in obj_list.keys():
        obj_list[key] = np.array(obj_list[key]) 
    # obj_list - keys - 'HE', 'DAPI', 'weight', 'group' - array containg individual files [1,2,3]
    if test_percentage != 0:
        skf = StratifiedShuffleSplit(n_splits=num, test_size=test_percentage / 100.0)
        for i_num, (train_idx, test_idx) in enumerate(skf.split(obj_list['HE'], groups_label)):
            cv_folder = os.path.join(flags['experiment_folder'], 'perm' + str(i_num + 1))
            create_dir(cv_folder)

            test_obj_list_dict = dict()
            train_obj_list_dict = dict()
            for key in obj_list.keys():
                test_obj_list_dict[key] = obj_list[key][test_idx]
                train_obj_list_dict[key] = obj_list[key][train_idx]
            
			# test_obj_list_dict - same as obj_list but only test files [3]
			# train_obj_list_dict - same as obj_list but only test files [1,2]
            train_groups_label = groups_label[train_idx]

            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_percentage / 100.0)
            for train_train_index, train_val_index in sss.split(train_obj_list_dict['HE'], train_groups_label):
                train_train_obj_list_dict = dict()
                train_val_obj_list_dict = dict()
                for key in train_obj_list_dict.keys():
                    train_train_obj_list_dict[key] = train_obj_list_dict[key][train_train_index]
                    train_val_obj_list_dict[key] = train_obj_list_dict[key][train_val_index]

            #################################################################
            # test
            for key in test_obj_list_dict.keys():
                filename = os.path.join(cv_folder, 'test_' + key + '_list.csv')
                if not os.path.isfile(filename):
                    row_list = [[item] for item in test_obj_list_dict[key]]
                    KScsv.write_csv(row_list, filename)

            #################################################################
            # train
            dict_path = flags['dict_path']
            dict_ext = flags['dict_ext']

            obj_list_dict = dict()
            for key in dict_path.keys():
                obj_list_dict[key] = glob.glob(os.path.join(dict_path[key], '*' + dict_ext[key]))

            temp_train_train_obj_list_dict = collections.defaultdict(list)

            for name in train_train_obj_list_dict['HE']:
                basename = os.path.basename(name)
                basename = os.path.splitext(basename)[0]
                matching = sorted([s for s in obj_list_dict['HE'] if basename in s])

                for m in matching:
                    basename = os.path.basename(m)
                    basename = os.path.splitext(basename)[0]

                    basename_dict = dict()
                    for key in train_train_obj_list_dict.keys():
                        basename_dict[key] = os.path.join(dict_path[key], basename + dict_ext[key])

                    if all(basename_dict[k] in obj_list_dict[k] for k in basename_dict.keys()):
                        for key in train_train_obj_list_dict.keys():
                            temp_train_train_obj_list_dict[key].append(basename_dict[key])

            for key in train_train_obj_list_dict.keys():
                train_train_obj_list_dict[key] = np.array(temp_train_train_obj_list_dict[key])

                filename = os.path.join(cv_folder, 'train_' + key + '_list.csv')
                if not os.path.isfile(filename):
                    row_list = [[item] for item in train_train_obj_list_dict[key]]
                    KScsv.write_csv(row_list, filename)

            #################################################################
            # validation
            dict_path = flags['dict_path']
            dict_ext = flags['dict_ext']

            obj_list_dict = dict()
            for key in dict_path.keys():
                obj_list_dict[key] = glob.glob(os.path.join(dict_path[key], '*' + dict_ext[key]))

            temp_train_val_obj_list_dict = collections.defaultdict(list)

            for name in train_val_obj_list_dict['HE']:
                basename = os.path.basename(name)
                basename = os.path.splitext(basename)[0]
                matching = sorted([s for s in obj_list_dict['HE'] if basename in s])

                for m in matching:
                    basename = os.path.basename(m)
                    basename = os.path.splitext(basename)[0]

                    basename_dict = dict()
                    for key in train_val_obj_list_dict.keys():
                        basename_dict[key] = os.path.join(dict_path[key], basename + dict_ext[key])

                    if all(basename_dict[k] in obj_list_dict[k] for k in basename_dict.keys()):
                        for key in train_val_obj_list_dict.keys():
                            temp_train_val_obj_list_dict[key].append(basename_dict[key])

            for key in train_val_obj_list_dict.keys():
                train_val_obj_list_dict[key] = np.array(temp_train_val_obj_list_dict[key])

                filename = os.path.join(cv_folder, 'val_' + key + '_list.csv')
                if not os.path.isfile(filename):
                    row_list = [[item] for item in train_val_obj_list_dict[key]]
                    KScsv.write_csv(row_list, filename)

    else:
        for i_num in range(num):
            cv_folder = os.path.join(flags['experiment_folder'], 'perm' + str(i_num + 1))
            create_dir(cv_folder)

            train_obj_list_dict = dict()
            for key in obj_list.keys():
                train_obj_list_dict[key] = obj_list[key]
            train_groups_label = groups_label

            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_percentage / 100.0)
            for train_train_index, train_val_index in sss.split(train_obj_list_dict['HE'], train_groups_label):
                train_train_obj_list_dict = dict()
                train_val_obj_list_dict = dict()
                for key in train_obj_list_dict.keys():
                    train_train_obj_list_dict[key] = train_obj_list_dict[key][train_train_index]
                    train_val_obj_list_dict[key] = train_obj_list_dict[key][train_val_index]

            #################################################################
            # train
            dict_path = flags['dict_path']
            dict_ext = flags['dict_ext']

            obj_list_dict = dict()
            for key in dict_path.keys():
                obj_list_dict[key] = glob.glob(os.path.join(dict_path[key], '*' + dict_ext[key]))

            temp_train_train_obj_list_dict = collections.defaultdict(list)

            for name in train_train_obj_list_dict['HE']:
                basename = os.path.basename(name)
                basename = os.path.splitext(basename)[0]
                matching = sorted([s for s in obj_list_dict['HE'] if basename in s])

                for m in matching:
                    basename = os.path.basename(m)
                    basename = os.path.splitext(basename)[0]

                    basename_dict = dict()
                    for key in train_train_obj_list_dict.keys():
                        basename_dict[key] = os.path.join(dict_path[key], basename + dict_ext[key])

                    if all(basename_dict[k] in obj_list_dict[k] for k in basename_dict.keys()):
                        for key in train_train_obj_list_dict.keys():
                            temp_train_train_obj_list_dict[key].append(basename_dict[key])

            for key in train_train_obj_list_dict.keys():
                train_train_obj_list_dict[key] = np.array(temp_train_train_obj_list_dict[key])

                filename = os.path.join(cv_folder, 'train_' + key + '_list.csv')
                if not os.path.isfile(filename):
                    row_list = [[item] for item in train_train_obj_list_dict[key]]
                    KScsv.write_csv(row_list, filename)

            #################################################################
            # validation

            dict_path = flags['dict_path']
            dict_ext = flags['dict_ext']

            obj_list_dict = dict()
            for key in dict_path.keys():
                obj_list_dict[key] = glob.glob(os.path.join(dict_path[key], '*' + dict_ext[key]))

            temp_train_val_obj_list_dict = collections.defaultdict(list)

            for name in train_val_obj_list_dict['HE']:
                basename = os.path.basename(name)
                basename = os.path.splitext(basename)[0]
                matching = sorted([s for s in obj_list_dict['image'] if basename in s])

                for m in matching:
                    basename = os.path.basename(m)
                    basename = os.path.splitext(basename)[0]

                    basename_dict = dict()
                    for key in train_val_obj_list_dict.keys():
                        basename_dict[key] = os.path.join(dict_path[key], basename + dict_ext[key])

                    if all(basename_dict[k] in obj_list_dict[k] for k in basename_dict.keys()):
                        for key in train_val_obj_list_dict.keys():
                            temp_train_val_obj_list_dict[key].append(basename_dict[key])

            for key in train_val_obj_list_dict.keys():
                train_val_obj_list_dict[key] = np.array(temp_train_val_obj_list_dict[key])

                filename = os.path.join(cv_folder, 'val_' + key + '_list.csv')
                if not os.path.isfile(filename):
                    row_list = [[item] for item in train_val_obj_list_dict[key]]
                    KScsv.write_csv(row_list, filename)


################################################################################################
def split_data(flags):
    obj_list = get_pair_list(flags['dict_path'], flags['dict_ext'])

    for key in obj_list.keys():
        tmp_list = list()
        for name in obj_list[key]:
            if flags['augmentation_keyword'] not in name:
                tmp_list.append(name)
        obj_list[key] = tmp_list

    # cross validation
    if flags['split_method'] == 'cv':
        split_cv(obj_list, flags)

    # permutation
    elif flags['split_method'] == 'perm':
        split_perm(obj_list,flags)

    else:
        raise ValueError('please select cv or perm')


################################################################################################
def gen_train_val_data(nth_fold, flags):
    """
    gen_train_val_data generate training and validation data for training the network. It build
    directories for train and test and extract patches according to the provided 'method'. It also keeps the log file
    :param nth_fold:
    :param method: 'sliding_window'
    :return:
    """
    # check whether 'cv' or 'perm' exists and which one to use ######################
    list_dir = os.listdir(os.path.join(flags['experiment_folder']))
    if ('cv' + str(nth_fold) in list_dir) and ('perm' + str(nth_fold) in list_dir):
        raise ValueError('Dangerous! You have both cv and perm on the path.')
    elif 'cv' + str(nth_fold) in list_dir:
        object_folder = os.path.join(flags['experiment_folder'], 'cv' + str(nth_fold))
    elif 'perm' + str(nth_fold) in list_dir:
        object_folder = os.path.join(flags['experiment_folder'], 'perm' + str(nth_fold))
    else:
        raise ValueError('No cv or perm folder!')

    # create checkpoint folder
    create_dir(os.path.join(object_folder,'checkpoint'))

    # create train and val paths ###############################
    path_dict = dict()
    path_dict['train_folder'] = os.path.join(object_folder, 'train')
    path_dict['val_folder'] = os.path.join(object_folder, 'val')
    create_dir(path_dict['train_folder'])
    create_dir(path_dict['val_folder'])

    # extract patches and put in a designated directory #################
    if flags['gen_train_val_method'] == 'sliding_window':

        #key_list = ['HE', 'DAPI', 'weight']
        key_list = list(flags['dict_path'].keys()) # all items except last item: 'group'
        key_list.remove('group')

        for key in key_list:
            path_dict['train_' + key + '_folder'] = os.path.join(path_dict['train_folder'], key)
            create_dir(path_dict['train_' + key + '_folder'])
            path_dict['val_' + key + '_folder'] = os.path.join(path_dict['val_folder'], key)
            create_dir(path_dict['val_' + key + '_folder'])

        list_dict = dict()
        for key in key_list:
            list_dict['train_' + key + '_list'] = KScsv.read_csv(
                os.path.join(object_folder, 'train_' + key + '_list.csv'))
            list_dict['val_' + key + '_list'] = KScsv.read_csv(os.path.join(object_folder, 'val_' + key + '_list.csv'))
        # for example list_dict['train_HE_list']
        # train ###########################################################
        for mode in ['train','val']:
            if not os.path.isfile(os.path.join(path_dict[mode + '_folder'], mode + '_log.csv')):
                log_data = list()

                for i_image in range(len(list_dict[mode + '_'+ key_list[0] +'_list'])):  # for range in HE_list, other folders should have same number of files as HE
                                                                                           # len -1 to exclude last entry in list_dict which is always empty
                    tic = time.time()
					
					#create dictionaries
                    paths = {}
                    dict_obj = {}
					
					# contain paths to various folders
                    for key in key_list:
                        paths[key] = list_dict[mode + '_'+ key +'_list'][i_image][0]
                        #images[key] = KSimage.imread(paths[key])
                        dict_obj[key] = KSimage.imread(paths[key])
						
                    #path_image = list_dict[mode + '_HE_list'][i_image][0]
                    #path_groundtruth = list_dict[mode + '_DAPI_list'][i_image][0]
                    #path_weight = list_dict[mode + '_weight_list'][i_image][0]

                    #image = KSimage.imread(path_image)
                    #groundtruth = KSimage.imread(path_groundtruth)
                    #weight = KSimage.imread(path_weight)

                    #dict_obj = {'image': image,
                    #            'groundtruth': groundtruth,
                    #            'weight':weight}

                    extractor = extract_patches.sliding_window(
                                                    dict_obj, flags['size_input_patch'], flags['size_output_patch'],flags['stride'])

                    
                    for j, (out_obj_dict, coord_dict) in enumerate(extractor):
                        coord_images = coord_dict[key_list[0]]   #{'HE': (0, 0), 'DAPI': (0, 0), 'weight': (0, 0)}
                        basename = os.path.basename(paths[key_list[0]])  # paths for HE
                        basename = os.path.splitext(basename)[0]      #name of img
                    
                        nametuple = ()

                        # Remove images with intensity greater than some amount - indicating large amounts of white space
                        HE_patch = out_obj_dict['HE']
                        if np.mean(HE_patch,axis=(0,1,2)) > flags['HE_thresh']:
                            continue
                       
                        for key in key_list:
                            image = out_obj_dict[key]
                            #groundtruths = out_obj_dict['groundtruth']
                            #weights = out_obj_dict['weight']

							#############################################################
                            file_name = os.path.join(path_dict[mode + '_' + key + '_folder'],
                                            basename + '_idx' + str(j) + '_row' + str(
                                                         coord_images[0]) + '_col' + str(coord_images[1]) + flags[key + '_ext'])

							#image_name = os.path.join(path_dict[mode + '_image_folder'],
						    #							  basename + '_idx' + str(j) + '_row' + str(
							#								  coord_images[0]) + '_col' + str(coord_images[1]) + flags['image_ext'])
						    #two levels of checking whether or not to write files
                            if not os.path.isfile(file_name):
                                KSimage.imwrite(image, file_name)
							#if not os.path.isfile(image_name):
							#	KSimage.imwrite(images, image_name)

                       
                            # count file in correspnding class
                            if mode == 'train':
                                 img_class = np.int32(file_name.split('/')[-1].split('_')[0])
                                 flags['class_counts'][img_class] += 1 
                               
                            # add to nametuple 
                            nametuple += (file_name,)
						
                        log_data.append(nametuple)

                    print('finish processing %d image from %d images : %.2f' % (
                        i_image + 1, len(list_dict[mode + '_' + key_list[0] + '_list']), time.time() - tic))

                    #counts
                    if mode == 'train':
                         countspath = os.path.join(os.path.join(object_folder,'checkpoint','counts.mat'))
                         matlab.save(countspath, {'counts': flags['class_counts']})

                    #csv
                    KScsv.write_csv(log_data, os.path.join(path_dict[mode + '_folder'], mode + '_log.csv'))

    ####################################################################################################################
    elif flags['gen_train_val_method'] == 'sliding_window_mask_single_prediction':

        key_list = ['image', 'groundtruth', 'tissue']

        for key in key_list:
            path_dict['train_' + key + '_folder'] = os.path.join(path_dict['train_folder'], key)
            create_dir(path_dict['train_' + key + '_folder'])
            path_dict['val_' + key + '_folder'] = os.path.join(path_dict['val_folder'], key)
            create_dir(path_dict['val_' + key + '_folder'])

        list_dict = dict()
        for key in key_list:
            list_dict['train_' + key + '_list'] = KScsv.read_csv(
                os.path.join(object_folder, 'train_' + key + '_list.csv'))
            list_dict['val_' + key + '_list'] = KScsv.read_csv(os.path.join(object_folder, 'val_' + key + '_list.csv'))


        # train ###########################################################
        for key in ['train','val']:
            if not os.path.isfile(os.path.join(path_dict[key + '_folder'], key + '_log.csv')):
                log_data = list()

                for i_image in range(len(list_dict[key + '_image_list'])):

                    tic = time.time()

                    path_image = list_dict[key + '_image_list'][i_image][0]
                    path_groundtruth = list_dict[key + '_groundtruth_list'][i_image][0]
                    path_tissue = list_dict[key + '_tissue_list'][i_image][0]

                    image = KSimage.imread(path_image)
                    groundtruth = KSimage.imread(path_groundtruth)
                    tissue = KSimage.imread(path_tissue)

                    dict_obj = {'image': image,
                                'groundtruth': groundtruth,
                                'tissue':tissue}

                    extractor = extract_patches.sliding_window(
                                                    dict_obj, flags['size_input_patch'], flags['size_output_patch'],flags['stride'])

                    for j, (out_obj_dict, coord_dict) in enumerate(extractor):
                        images = out_obj_dict['image']
                        groundtruths = out_obj_dict['groundtruth']
                        tissues = out_obj_dict['tissue']
                        coord_images = coord_dict['image']

                        #############################################################

                        groundtruth_area = np.logical_and(tissues == 255.0, groundtruths == 255.0)
                        true_class = np.sum(groundtruth_area) / float(groundtruths.size)
                        non_groundtruth_area = np.logical_and(tissues == 255.0, np.logical_not(groundtruths == 255.0))
                        neg_class = np.sum(non_groundtruth_area) / float(groundtruths.size)
                        if true_class > 0.8:
                            labels = 1
                        elif neg_class > 0.8:
                            labels = 0
                        else:
                            labels = -1

                        #############################################################
                        basename = os.path.basename(path_image)
                        basename = os.path.splitext(basename)[0]

                        if labels != -1:
                            image_name = os.path.join(path_dict[key + '_image_folder'],
                                                      basename + '_idx' + str(j) + '_row' + str(
                                                          coord_images[0]) + '_col' + str(coord_images[1]) + flags['image_ext'])
                            label_name = os.path.join(path_dict[key + '_groundtruth_folder'],
                                                      basename + '_idx' + str(j) + '_row' + str(
                                                          coord_images[0]) + '_col' + str(coord_images[1]) + flags['label_ext'])

                            if not os.path.isfile(image_name):
                                KSimage.imwrite(images, image_name)

                            if not os.path.isfile(label_name):
                                KScsv.write_csv(str(labels), label_name)

                            log_data.append((image_name, label_name, str(labels)))

                    print('finish processing %d image from %d images : %.2f' % (
                        i_image + 1, len(list_dict[key + '_image_list']), time.time() - tic))

                KScsv.write_csv(log_data, os.path.join(path_dict[key + '_folder'], key + '_log.csv'))

    ####################################################################################################################
    elif flags['gen_train_val_method'] == 'sliding_window_mask_single_prediction_centre':

        key_list = ['image', 'groundtruth']

        for key in key_list:
            path_dict['train_' + key + '_folder'] = os.path.join(path_dict['train_folder'], key)
            create_dir(path_dict['train_' + key + '_folder'])
            path_dict['val_' + key + '_folder'] = os.path.join(path_dict['val_folder'], key)
            create_dir(path_dict['val_' + key + '_folder'])

        list_dict = dict()
        for key in key_list:
            list_dict['train_' + key + '_list'] = KScsv.read_csv(
                os.path.join(object_folder, 'train_' + key + '_list.csv'))
            list_dict['val_' + key + '_list'] = KScsv.read_csv(
                os.path.join(object_folder, 'val_' + key + '_list.csv'))

        # train ###########################################################
        for key in ['train', 'val']:
            if not os.path.isfile(os.path.join(path_dict[key + '_folder'], key + '_log.csv')):
                log_data = list()

                for i_image in range(len(list_dict[key + '_image_list'])):

                    tic = time.time()

                    path_image = list_dict[key + '_image_list'][i_image][0]
                    path_groundtruth = list_dict[key + '_groundtruth_list'][i_image][0]

                    image = KSimage.imread(path_image)
                    groundtruth = KSimage.imread(path_groundtruth)

                    dict_obj = {'image': image,
                                'groundtruth': groundtruth}

                    extractor = extract_patches.sliding_window(
                        dict_obj, flags['size_input_patch'], flags['size_output_patch'], flags['stride'])

                    for j, (out_obj_dict, coord_dict) in enumerate(extractor):
                        images = out_obj_dict['image']
                        groundtruths = out_obj_dict['groundtruth']
                        coord_images = coord_dict['image']

                        #############################################################
                        if groundtruths[int(groundtruths.shape[0]/2.0),int(groundtruths.shape[1]/2.0)] == 255.0:
                            labels = 1
                        else:
                            labels = 0

                        #############################################################
                        basename = os.path.basename(path_image)
                        basename = os.path.splitext(basename)[0]

                        if labels != -1:
                            image_name = os.path.join(path_dict[key + '_image_folder'],
                                                      basename + '_idx' + str(j) + '_row' + str(
                                                          coord_images[0]) + '_col' + str(coord_images[1]) +
                                                      flags['image_ext'])
                            label_name = os.path.join(path_dict[key + '_groundtruth_folder'],
                                                      basename + '_idx' + str(j) + '_row' + str(
                                                          coord_images[0]) + '_col' + str(coord_images[1]) +
                                                      flags['label_ext'])

                            if not os.path.isfile(image_name):
                                KSimage.imwrite(images, image_name)

                            if not os.path.isfile(label_name):
                                KScsv.write_csv(str(labels), label_name)

                            log_data.append((image_name, label_name, str(labels)))

                    print('finish processing %d image from %d images : %.2f' % (
                        i_image + 1, len(list_dict[key + '_image_list']), time.time() - tic))

                KScsv.write_csv(log_data, os.path.join(path_dict[key + '_folder'], key + '_log.csv'))

    ####################################################################################################################
    elif flags['gen_train_val_method'] == 'sliding_window_mask_single_prediction_centre_mask':

        key_list = ['image', 'groundtruth', 'mask']

        for key in key_list:
            path_dict['train_' + key + '_folder'] = os.path.join(path_dict['train_folder'], key)
            create_dir(path_dict['train_' + key + '_folder'])
            path_dict['val_' + key + '_folder'] = os.path.join(path_dict['val_folder'], key)
            create_dir(path_dict['val_' + key + '_folder'])

        list_dict = dict()
        for key in key_list:
            list_dict['train_' + key + '_list'] = KScsv.read_csv(
                os.path.join(object_folder, 'train_' + key + '_list.csv'))
            list_dict['val_' + key + '_list'] = KScsv.read_csv(
                os.path.join(object_folder, 'val_' + key + '_list.csv'))

        # train ###########################################################
        for key in ['train', 'val']:
            if not os.path.isfile(os.path.join(path_dict[key + '_folder'], key + '_log.csv')):
                log_data = list()

                for i_image in range(len(list_dict[key + '_image_list'])):

                    tic = time.time()

                    path_image = list_dict[key + '_image_list'][i_image][0]
                    path_groundtruth = list_dict[key + '_groundtruth_list'][i_image][0]
                    path_mask = list_dict[key + '_mask_list'][i_image][0]

                    image = KSimage.imread(path_image)
                    groundtruth = KSimage.imread(path_groundtruth)
                    mask = KSimage.imread(path_mask)

                    dict_obj = {'image': image,
                                'groundtruth': groundtruth,
                                'mask': mask}

                    extractor = extract_patches.sliding_window(
                        dict_obj, flags['size_input_patch'], flags['size_output_patch'], flags['stride'])

                    for j, (out_obj_dict, coord_dict) in enumerate(extractor):
                        images = out_obj_dict['image']
                        groundtruths = out_obj_dict['groundtruth']
                        masks = out_obj_dict['mask']
                        coord_images = coord_dict['image']

                        #############################################################
                        if masks[int(groundtruths.shape[0] / 2.0), int(groundtruths.shape[1] / 2.0)] == 255.0:
                            do = True
                        else:
                            do = np.random.uniform(low=0.0, high=1.0, size=1) > 0.50

                        if do:

                            if groundtruths[
                                int(groundtruths.shape[0] / 2.0), int(groundtruths.shape[1] / 2.0)] == 255.0:
                                labels = 1
                            else:
                                labels = 0

                            #############################################################
                            basename = os.path.basename(path_image)
                            basename = os.path.splitext(basename)[0]

                            if labels != -1:
                                image_name = os.path.join(path_dict[key + '_image_folder'],
                                                          basename + '_idx' + str(j) + '_row' + str(
                                                              coord_images[0]) + '_col' + str(coord_images[1]) +
                                                          flags['image_ext'])
                                label_name = os.path.join(path_dict[key + '_groundtruth_folder'],
                                                          basename + '_idx' + str(j) + '_row' + str(
                                                              coord_images[0]) + '_col' + str(coord_images[1]) +
                                                          flags['label_ext'])

                                if not os.path.isfile(image_name):
                                    KSimage.imwrite(images, image_name)

                                if not os.path.isfile(label_name):
                                    KScsv.write_csv(str(labels), label_name)

                                log_data.append((image_name, label_name, str(labels)))

                    print('finish processing %d image from %d images : %.2f' % (
                        i_image + 1, len(list_dict[key + '_image_list']), time.time() - tic))

                KScsv.write_csv(log_data, os.path.join(path_dict[key + '_folder'], key + '_log.csv'))
    ####################################################################################################################
    elif flags['gen_train_val_method'] == 'coordinate':

        key_list = ['image', 'coordinate', 'label']

        for key in key_list:
            path_dict['train_' + key + '_folder'] = os.path.join(path_dict['train_folder'], key)
            create_dir(path_dict['train_' + key + '_folder'])
            path_dict['val_' + key + '_folder'] = os.path.join(path_dict['val_folder'], key)
            create_dir(path_dict['val_' + key + '_folder'])

        # read lists of images, labels, groups #########################
        list_dict = dict()
        for key in key_list:
            list_dict['train_' + key + '_list'] = KScsv.read_csv(
                os.path.join(object_folder, 'train_' + key + '_list.csv'))
            list_dict['val_' + key + '_list'] = KScsv.read_csv(os.path.join(object_folder, 'val_' + key + '_list.csv'))

        # train and val
        for key in ['train', 'val']:
            if not os.path.isfile(os.path.join(path_dict[key + '_folder'], key + '_log.csv')):
                log_data = list()

                for i_image in range(len(list_dict[key + '_image_list'])):

                    tic = time.time()

                    path_image = list_dict[key + '_image_list'][i_image][0]
                    path_label = list_dict[key + '_label_list'][i_image][0]
                    path_coordinate = list_dict[key + '_coordinate_list'][i_image][0]

                    image = KSimage.imread(path_image)
                    labels = KScsv.read_csv(path_label)
                    mat_content = matlab.load(path_coordinate)
                    coordinate = mat_content['coordinate']
                    coordinate = coordinate.astype(np.int32)

                    # pad array
                    padrow = flags['size_input_patch'][0] + 1
                    padcol = flags['size_input_patch'][1] + 1

                    if image.ndim == 2:
                        image = np.lib.pad(image, ((padrow, padrow), (padcol, padcol)), 'symmetric')
                    else:
                        image = np.lib.pad(image, ((padrow, padrow), (padcol, padcol), (0, 0)), 'symmetric')

                    shifted_coordinate = np.copy(coordinate)
                    shifted_coordinate[:, 0] += (padrow - 1)
                    shifted_coordinate[:, 1] += (padcol - 1)

                    dict_obj = {'image': image}
                    dict_patch_size = {'image': flags['size_input_patch']}

                    dict_obj_out = extract_patches.coordinate(dict_obj, dict_patch_size, shifted_coordinate)

                    for j, (dict_patches, coord_dict) in enumerate(dict_obj_out):

                        images = dict_patches['image']

                        image_basename = os.path.basename(path_image)
                        image_basename = os.path.splitext(image_basename)[0]

                        image_name = os.path.join(path_dict[key + '_image_folder'],
                                                  image_basename + '_idx' + str(j) + '_row' + str(
                                                      coord_dict['image'][0]) + '_col' + str(coord_dict['image'][1]) +
                                                  flags['image_ext'])
                        label_name = os.path.join(path_dict[key + '_label_folder'],
                                                  image_basename + '_idx' + str(j) + '_row' + str(
                                                      coord_dict['image'][0]) + '_col' + str(coord_dict['image'][1]) +
                                                  flags['label_ext'])
                        # weight_name = os.path.join(path_dict[key + '_weight_folder'],
                        #                            image_basename + '_idx' + str(j) + '_row' + str(
                        #                                coord_dict['image'][0]) + '_col' + str(
                        #                                coord_dict['image'][1]) + flags['weight_ext'])

                        if not os.path.isfile(image_name):
                            KSimage.imwrite(images, image_name)

                        if not os.path.isfile(label_name):
                            KScsv.write_csv(labels[j][0], label_name)

                        log_data.append((image_name, label_name, labels[j][0]))

                    print('finish processing %d image from %d images : %.2f' % (
                        i_image + 1, len(list_dict[key + '_image_list']), time.time() - tic))

                KScsv.write_csv(log_data, os.path.join(path_dict[key + '_folder'], key + '_log.csv'))

    ####################################################################################################################
    elif flags['gen_train_val_method'] == 'coordinate_jittering':

        key_list = ['image', 'coordinate', 'label']

        for key in key_list:
            path_dict['train_' + key + '_folder'] = os.path.join(path_dict['train_folder'], key)
            create_dir(path_dict['train_' + key + '_folder'])
            path_dict['val_' + key + '_folder'] = os.path.join(path_dict['val_folder'], key)
            create_dir(path_dict['val_' + key + '_folder'])

        # read lists of images, labels, groups #########################
        list_dict = dict()
        for key in key_list:
            list_dict['train_' + key + '_list'] = KScsv.read_csv(
                os.path.join(object_folder, 'train_' + key + '_list.csv'))
            list_dict['val_' + key + '_list'] = KScsv.read_csv(
                os.path.join(object_folder, 'val_' + key + '_list.csv'))

        # train and val
        for key in ['train', 'val']:
            if not os.path.isfile(os.path.join(path_dict[key + '_folder'], key + '_log.csv')):
                log_data = list()

                for i_image in range(len(list_dict[key + '_image_list'])):

                    tic = time.time()

                    path_image = list_dict[key + '_image_list'][i_image][0]
                    path_label = list_dict[key + '_label_list'][i_image][0]
                    path_coordinate = list_dict[key + '_coordinate_list'][i_image][0]

                    image = KSimage.imread(path_image)
                    labels = KScsv.read_csv(path_label)
                    mat_content = matlab.load(path_coordinate)
                    coordinate = mat_content['coordinate']
                    coordinate = coordinate.astype(np.int32)

                    # pad array
                    padrow = flags['size_input_patch'][0] + 1
                    padcol = flags['size_input_patch'][1] + 1

                    if image.ndim == 2:
                        image = np.lib.pad(image, ((padrow, padrow), (padcol, padcol)), 'symmetric')
                    else:
                        image = np.lib.pad(image, ((padrow, padrow), (padcol, padcol), (0, 0)), 'symmetric')

                    shifted_coordinate = np.copy(coordinate)
                    shifted_coordinate[:, 0] += (padrow - 1)
                    shifted_coordinate[:, 1] += (padcol - 1)

                    dict_obj = {'image': image}
                    dict_patch_size = {'image': flags['size_input_patch']}

                    for loop in range(flags['n_jittering']):
                        dict_obj_out = extract_patches.coordinate_jittering_with_radius(dict_obj,
                                       dict_patch_size, shifted_coordinate,flags['jittering_radius'])

                        for j, (dict_patches, coord_dict) in enumerate(dict_obj_out):

                            images = dict_patches['image']

                            image_basename = os.path.basename(path_image)
                            image_basename = os.path.splitext(image_basename)[0]

                            image_name = os.path.join(path_dict[key + '_image_folder'],
                                                    image_basename + '_idx' + str(j) + '_row' + str(
                                                        coord_dict['image'][0]) + '_col' + str(
                                                        coord_dict['image'][1]) +
                                                    flags['image_ext'])
                            label_name = os.path.join(path_dict[key + '_label_folder'],
                                                    image_basename + '_idx' + str(j) + '_row' + str(
                                                        coord_dict['image'][0]) + '_col' + str(
                                                        coord_dict['image'][1]) +
                                                    flags['label_ext'])

                            if not os.path.isfile(image_name):
                                KSimage.imwrite(images, image_name)

                            if not os.path.isfile(label_name):
                                KScsv.write_csv(labels[j][0], label_name)

                            log_data.append((image_name, label_name, labels[j][0]))

                    print('finish processing %d image from %d images : %.2f' % (
                        i_image + 1, len(list_dict[key + '_image_list']), time.time() - tic))

                KScsv.write_csv(log_data, os.path.join(path_dict[key + '_folder'], key + '_log.csv'))

    ####################################################################################################################
    elif flags['gen_train_val_method'] == 'detection':

        key_list = ['image', 'groundtruth', 'weight']

        for key in key_list:
            path_dict['train_' + key + '_folder'] = os.path.join(path_dict['train_folder'], key)
            create_dir(path_dict['train_' + key + '_folder'])
            path_dict['val_' + key + '_folder'] = os.path.join(path_dict['val_folder'], key)
            create_dir(path_dict['val_' + key + '_folder'])

        # read lists of images, labels, groups #########################
        list_dict = dict()
        for key in key_list:
            list_dict['train_' + key + '_list'] = KScsv.read_csv(
                os.path.join(object_folder, 'train_' + key + '_list.csv'))
            list_dict['val_' + key + '_list'] = KScsv.read_csv(os.path.join(object_folder, 'val_' + key + '_list.csv'))

        # train and val
        for key in ['train', 'val']:
            if not os.path.isfile(os.path.join(path_dict[key + '_folder'], key + '_log.csv')):
                log_data = list()

                for i_image in range(len(list_dict[key + '_image_list'])):

                    tic = time.time()

                    path_image = list_dict[key + '_image_list'][i_image][0]
                    path_label = list_dict[key + '_groundtruth_list'][i_image][0]
                    path_weight = list_dict[key + '_weight_list'][i_image][0]

                    image = KSimage.imread(path_image)
                    labels = KSimage.imread(path_label)
                    weights = KSimage.imread(path_weight)
                    groundtruth = (labels == 255.0).astype(np.int32)
                    regions = regionprops(label(groundtruth))

                    coordinate = []
                    for props in regions:
                        x0, y0 = props.centroid
                        coordinate.append([x0,y0])

                    coordinate = np.array(coordinate)
                    coordinate = coordinate.astype(np.int32)


                    # pad array
                    padrow = flags['size_input_patch'][0] + 1
                    padcol = flags['size_input_patch'][1] + 1

                    if image.ndim == 2:
                        image = np.lib.pad(image, ((padrow, padrow), (padcol, padcol)), 'symmetric')
                    else:
                        image = np.lib.pad(image, ((padrow, padrow), (padcol, padcol), (0, 0)), 'symmetric')

                    if labels.ndim == 2:
                        labels = np.lib.pad(labels, ((padrow, padrow), (padcol, padcol)), 'symmetric')
                    else:
                        labels = np.lib.pad(labels, ((padrow, padrow), (padcol, padcol), (0, 0)), 'symmetric')

                    if weights.ndim == 2:
                        weights = np.lib.pad(weights, ((padrow, padrow), (padcol, padcol)), 'symmetric')
                    else:
                        weights = np.lib.pad(weights, ((padrow, padrow), (padcol, padcol), (0, 0)), 'symmetric')

                    dict_obj = {'image': image, 'groundtruth': labels, 'weight': weights}
                    dict_patch_size = {'image': flags['size_input_patch'], 'groundtruth': flags['size_output_patch'],
                                       'weight': flags['size_output_patch']}

                    if coordinate.size != 0:
                        shifted_coordinate = np.copy(coordinate)
                        shifted_coordinate[:, 0] += (padrow - 1)
                        shifted_coordinate[:, 1] += (padcol - 1)

                        for loop in range(flags['n_jittering']):
                            dict_obj_out = extract_patches.coordinate_jittering(dict_obj, dict_patch_size, shifted_coordinate)

                            for j, (dict_patches, coord_dict) in enumerate(dict_obj_out):

                                images = dict_patches['image']
                                labels = dict_patches['groundtruth']
                                weights = dict_patches['weight']

                                image_basename = os.path.basename(path_image)
                                image_basename = os.path.splitext(image_basename)[0]

                                image_name = os.path.join(path_dict[key + '_image_folder'],
                                                          image_basename + '_idx' + str(j) + '_row' + str(
                                                      coord_dict['image'][0]) + '_col' + str(coord_dict['image'][1]) + flags['image_ext'])
                                label_name = os.path.join(path_dict[key + '_groundtruth_folder'],
                                                          image_basename + '_idx' + str(j) + '_row' + str(
                                                      coord_dict['image'][0]) + '_col' + str(coord_dict['image'][1]) + flags['groundtruth_ext'])
                                weight_name = os.path.join(path_dict[key + '_weight_folder'],
                                                           image_basename + '_idx' + str(j) + '_row' + str(
                                                              coord_dict['image'][0]) + '_col' + str(
                                                              coord_dict['image'][1]) + flags['weight_ext'])

                                if not os.path.isfile(image_name):
                                    KSimage.imwrite(images, image_name)

                                if not os.path.isfile(label_name):
                                    KSimage.imwrite(labels, label_name)

                                if not os.path.isfile(weight_name):
                                    KSimage.imwrite(weights, weight_name)
                                log_data.append((image_name, label_name, weight_name))

                    # background area
                    extractor = extract_patches.sliding_window(
                        dict_obj, flags['size_input_patch'], flags['size_output_patch'], flags['stride'])

                    for j, (out_obj_dict, coord_dict) in enumerate(extractor):
                        images = out_obj_dict['image']
                        groundtruths = out_obj_dict['groundtruth']
                        weights = out_obj_dict['weight']

                        coord_images = coord_dict['image']

                        #############################################################

                        basename = os.path.basename(path_image)
                        basename = os.path.splitext(basename)[0]

                        image_name = os.path.join(path_dict[key + '_image_folder'],
                                                  basename + '_idx' + str(j) + '_row' + str(
                                                      coord_images[0]) + '_col' + str(
                                                      coord_images[1]) + flags['image_ext'])
                        label_name = os.path.join(path_dict[key + '_groundtruth_folder'],
                                                  basename + '_idx' + str(j) + '_row' + str(
                                                      coord_images[0]) + '_col' + str(
                                                      coord_images[1]) + flags['groundtruth_ext'])
                        weight_name = os.path.join(path_dict[key + '_weight_folder'],
                                                   basename + '_idx' + str(j) + '_row' + str(
                                                      coord_images[0]) + '_col' + str(
                                                      coord_images[1]) + flags['weight_ext'])

                        if not os.path.isfile(image_name):
                            KSimage.imwrite(images, image_name)

                        if not os.path.isfile(label_name):
                            KSimage.imwrite(groundtruths, label_name)

                        if not os.path.isfile(weight_name):
                            KSimage.imwrite(weights, weight_name)

                        log_data.append((image_name, label_name, weight_name))

                    print('finish processing %d image from %d images : %.2f' % (
                            i_image + 1, len(list_dict[key + '_image_list']), time.time() - tic))

                KScsv.write_csv(log_data, os.path.join(path_dict[key + '_folder'], key + '_log.csv'))

    ####################################################################################################################
    else:
        raise ValueError('no method selected!')


################################################################################################
def select_train_val_instances(nth_fold, method, flags):
    """
    select_train_val_instances is used to balance the class instances
    :param nth_fold:
    :param method:
    :return:
    """
    # check if log files exist
    list_dir = os.listdir(os.path.join(flags['experiment_folder']))
    if ('cv' + str(nth_fold) in list_dir) and ('perm' + str(nth_fold) in list_dir):
        raise ValueError('Dangerous! You have both cv and perm on the path.')
    elif 'cv' + str(nth_fold) in list_dir:
        object_folder = os.path.join(flags['experiment_folder'], 'cv' + str(nth_fold))
    elif 'perm' + str(nth_fold) in list_dir:
        object_folder = os.path.join(flags['experiment_folder'], 'perm' + str(nth_fold))
    else:
        raise ValueError('No cv or perm folder!')

    train_log_file_path = os.path.join(object_folder, 'train', 'train_log.csv')
    val_log_file_path = os.path.join(object_folder, 'val', 'val_log.csv')

    if not os.path.isfile(train_log_file_path):
        raise ValueError('no ' + train_log_file_path)
    if not os.path.isfile(val_log_file_path):
        raise ValueError('no ' + val_log_file_path)

    # read csv
    train_log = KScsv.read_csv(train_log_file_path)
    val_log = KScsv.read_csv(val_log_file_path)

    # count the number
    if method == 'by_numbers':
        train_log = select_instances.by_numbers(train_log)
        val_log = select_instances.by_numbers(val_log)

        KScsv.write_csv(train_log, train_log_file_path)
        KScsv.write_csv(val_log, val_log_file_path)
    else:
        raise ValueError('no method ' + method + ' exists!')


################################################################################################
def gen_weight(choice, flags):
    if choice == 'cell_segmentation':
        # get ground truth images
        list_gt = glob.glob(os.path.join(flags['annotation_groundtruths_folder'], '*' + flags['label_ext']))

        # create weights directory
        create_dir(flags['annotation_weights_folder'])

        # loop throught label images
        for file in list_gt:

            basename = os.path.basename(file)
            basename = os.path.splitext(basename)[0]
            savename = os.path.join(flags['annotation_weights_folder'], basename + flags['label_ext'])

            # check if the file exists
            if not os.path.isfile(savename):

                # imread
                I = KSimage.imread(file)

                # convert to [0,1]
                I = I.astype('float')
                I = I / 255.0

                # get distance transform
                mask = I > 0.5
                mask = KSimage.bwperim(mask)

                bw_labels = KSimage.bwlabel(mask)
                max_idx = np.max(bw_labels)
                if max_idx == 0:
                    weight = np.zeros((bw_labels.shape[0], bw_labels.shape[1]), dtype=np.float)
                else:
                    D = np.zeros((bw_labels.shape[0], bw_labels.shape[1], max_idx), dtype=np.float)
                    for idx in range(max_idx):
                        mask = bw_labels == idx + 1
                        D[:, :, idx] = KSimage.distance_transform(mask)

                    D = np.sort(D, axis=2)
                    if D.shape[2] > 1:
                        D = D[:, :, 0] + D[:, :, 1]
                    else:
                        D *= 2.0

                    weight = np.exp(-np.square(D) / np.square(flags['sigma']))

                weight *= 255.0
                weight = weight.astype(np.uint8)
                weight = np.squeeze(weight)

                KSimage.imwrite(weight, savename)
    elif choice == 'tumour_segmentation':
        # get ground truth images
        list_gt = glob.glob(os.path.join(flags['annotation_groundtruths_folder'], '*' + flags['label_ext']))

        # create weights directory
        create_dir(flags['annotation_weights_folder'])

        # loop throught label images
        for file in list_gt:

            basename = os.path.basename(file)
            basename = os.path.splitext(basename)[0]
            savename = os.path.join(flags['annotation_weights_folder'], basename + flags['label_ext'])

            # check if the file exists
            if not os.path.isfile(savename):
                # imread
                I = KSimage.imread(file)

                weight = np.ones((I.shape[0], I.shape[1])) * 255.0
                weight = weight.astype(np.uint8)
                weight = np.squeeze(weight)

                KSimage.imwrite(weight, savename)
    else:
        print("no choice for %s" % (choice))
        raise ValueError('terminate!')


################################################################################################
def fish2dapi(tiff_image):
    path, filename = os.path.split(tiff_image)
    basename = os.path.splitext(filename)[0]
    savename = os.path.join(path, basename + '.png')

    if not os.path.isfile(savename):
        I = KSimage.imread(tiff_image)
        B = I[:, :, 2]
        KSimage.imwrite(B, savename)


################################################################################################
def convert_fish_2_dapi(test_image_path):
    # file_list = KScsv.read_csv(test_image_path)
    file_list = test_image_path
    for iImage, file in enumerate(file_list):
        tic = time.time()
        fish2dapi(file)
        duration = time.time() - tic
        print('process %d / %d images (%.2f sec)' % (iImage + 1, len(file_list), duration))


################################################################################################
def retouch_segmentation(file):
    matcontent = matlab.load(file)
    mask = matcontent['mask']
    mask = np.squeeze(mask)

    # threshold
    binary_mask = mask > 0.8
    binary_mask_base = mask > 0.5

    # define disk structure
    radius = 5
    [x, y] = np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1))
    z = np.sqrt(x ** 2 + y ** 2)
    structure = z < radius

    # imerosion
    erode_mask = binary_erosion(binary_mask, structure=structure, border_value=1)
    erode_mask = remove_small_objects(erode_mask, 100)

    # watershed
    distance = ndimage.distance_transform_edt(binary_mask_base)
    markers = ndimage.label(erode_mask)[0]
    labels = watershed(-distance, markers, mask=binary_mask_base)

    return labels


################################################################################################
def post_processing_segmentation(test_image_path):
    # file_list = KScsv.read_csv(test_image_path)
    file_list = test_image_path

    post_process_folder = os.path.join('postprocess')
    create_dir(post_process_folder)

    for iImage, file in enumerate(file_list):
        tic = time.time()

        path, filename = os.path.split(file)
        basename = os.path.splitext(filename)[0]
        savename = os.path.join(post_process_folder, basename + '.mat')

        if not os.path.isfile(savename):
            labels = retouch_segmentation(file)
            matlab.save(savename, {'mask': labels})

        duration = time.time() - tic
        print('process %d / %d images (%.2f sec)' % (iImage + 1, len(file_list), duration))


################################################################################################
def segment_non_tissue_bg(img):
    if img.ndim != 2:
        img = KSimage.rgb2gray(img)
        # raise ValueError('not gray image!')

    temp = np.copy(img)
    temp = np.reshape(temp, -1)
    temp = temp[temp != 0]

    # global otsu
    threshold_global_otsu = threshold_otsu(temp)
    global_otsu = img >= threshold_global_otsu

    # remove small object
    open_area = remove_small_objects(global_otsu, min_size=50 ** 2)

    # convex hull
    try:
        convex_hull = convex_hull_image(open_area)
    except:
        convex_hull = open_area

    # dilation
    selem = disk(20)
    dilated_hull = rank.maximum(convex_hull, selem)

    return dilated_hull


################################################################################################
def segment_tissue_area(test_image_path,folder):
    file_list = test_image_path

    for iImage, file in enumerate(file_list):
        tic = time.time()

        path, filename = os.path.split(file)
        basename = os.path.splitext(filename)[0]
        savename = os.path.join(folder, basename + '.png')

        if not os.path.isfile(savename):
            img = KSimage.imread(file)
            mask = segment_non_tissue_bg(img)
            KSimage.imwrite(mask, savename)

        duration = time.time() - tic
        print('process %d / %d images (%.2f sec)' % (iImage + 1, len(file_list), duration))

################################################################################################
def register_image_translation(img1,img2):
    if img1.ndim != 2:
        img1_gray = KSimage.rgb2gray(img1)
    if img2.ndim != 2:
        img2_gray = KSimage.rgb2gray(img2)

    shift, error, diffphase = register_translation(img1_gray, img2_gray, 100)
    tform = SimilarityTransform(translation=(-shift[1], -shift[0]))
    warped = warp(img2, tform)

    return warped
