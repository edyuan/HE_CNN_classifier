# import libraries and packages
from KS_lib.general import KScsv
import os
import glob
import collections
import re
import numpy as np
from skimage.morphology import watershed
import cv2
from scipy import ndimage
from scipy import stats
from scipy.spatial.distance import cdist
import time
import multiprocessing
import psutil
import subprocess
from Modules import notebook_util

from KS_lib.general import matlab
from KS_lib.image import KSimage
from KS_lib.prepare_data import routine
from sklearn.neighbors import NearestNeighbors
# from KS_lib.tf_model_he_dcis_segmentation import tf_model_main as main_he_dcis_segmentation
# from KS_lib.tf_model_he_cell_segmentation import tf_model_main as main_he_cell_segmentation
# from KS_lib.tf_model_dcis_cell_segmentation import tf_model_main as main_dcis_cell_segmentation
# from KS_lib.tf_model_probe_detection import tf_model_main as main_probe_detection
# from Modules.flags_he_dcis_segmentation import flags as flags_he_dcis_segmentation
# from Modules.flags_he_cell_segmentation import flags as flags_he_cell_segmentation
# from Modules.flags_dcis_cell_segmentation import flags as flags_dcis_cell_segmentation
# from Modules.flags_probe_detection_green import flags as flags_probe_detection_green
# from Modules.flags_probe_detection_red import flags as flags_probe_detection_red

from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


###############################################
def get_pair_list(dict_path, dict_ext):
    images_list = dict()
    for key in dict_path.keys():
        images_list[key] = glob.glob(os.path.join(dict_path[key], '*' + dict_ext[key]))

    obj_list = collections.defaultdict(list)

    for image_name in images_list[dict_path.keys()[0]]:
        basename = os.path.basename(image_name)
        basename = os.path.splitext(basename)[0]
        pos = [m.start() for m in re.finditer('_', basename)]
        # basename = basename[0:pos[3]+1]

        dict_name = dict()
        for key in dict_path.keys():
            filename = [x for x in images_list[key] if basename in x]
            if filename:
                dict_name[key] = filename[0]
            else:
                dict_name[key] = 'none'

        if all(os.path.isfile(v) for k, v in dict_name.items()):
            for key in dict_path.keys():
                obj_list[key].append(dict_name[key])

    for key in obj_list.keys():
        if not obj_list[key]:
            print("no data in %s" % (dict_path[key]))
            raise ValueError('terminate!')

    return obj_list


###############################################
def write_log_file(dict_path, dict_ext, log_filename):
    # check if the directories exist
    for key in dict_path.keys():
        assert os.path.exists(dict_path[key]), '%s directory does not exist' % (dict_path[key])

    # generate a log file for samples that have a complete set of 3 types of images

    obj_list = get_pair_list(dict_path, dict_ext)

    row_list = list()
    for iimage in range(len(obj_list[obj_list.keys()[0]])):
        temp = list()
        for key in obj_list.keys():
            temp.append(obj_list[key][iimage])

        row_list.append(temp)

    KScsv.write_csv(row_list, log_filename)


###############################################
def para_preprocess(args):
    row, i, len_log_file, dict_path_preproc, dict_ext_preproc, dict_type = args
    t = time.time()

    basename = os.path.split(row)[1]
    basename = os.path.splitext(basename)[0]
    savename = os.path.join(dict_path_preproc, basename + dict_ext_preproc)

    if not os.path.isfile(savename):
        if dict_type == 'HE' or dict_type == 'IHC':
            img = KSimage.imread(row)
            KSimage.imwrite(img, savename)

        elif dict_type == 'FISH':
            img = KSimage.imread(row)
            hsv = KSimage.rgb2hsv(img)
            hsv[:, :, 2] = KSimage.adaptive_histeq(hsv[:, :, 2])
            rgb = KSimage.hsv2rgb(hsv)
            KSimage.imwrite(rgb, savename)

    duration = time.time() - t
    print('Finish preprocessing images from sample %d out of %d samples (%.2f sec)' % (i + 1, len_log_file, duration))


###############################################
def preprocess(dict_path, dict_ext, dict_path_preproc, dict_ext_preproc, dict_type):
    obj_list = get_pair_list(dict_path, dict_ext)

    # create a preprocess image directories
    for key in dict_path_preproc.keys():
        routine.create_dir(dict_path_preproc[key])

    # preprocess images according to their type
    # for i, row in enumerate(log_file):
    #     t = time.time()
    #     for j, img_filename in enumerate(row):
    #         basename = os.path.split(img_filename)[1]
    #         basename = os.path.splitext(basename)[0]
    #         savename = os.path.join(dict_path_preproc[dict_path_preproc.keys()[j]],
    #                                 basename + dict_ext_preproc[dict_ext_preproc.keys()[j]])
    #
    #         if not os.path.isfile(savename):
    #             if dict_type[dict_type.keys()[j]] == 'HE':
    #                 img = KSimage.imread(img_filename)
    #                 KSimage.imwrite(img,savename)
    #
    #             elif dict_type[dict_type.keys()[j]] == 'FISH':
    #                 img = KSimage.imread(img_filename)
    #                 hsv = KSimage.rgb2hsv(img)
    #                 hsv[:,:,2] = KSimage.adaptive_histeq(hsv[:,:,2])
    #                 rgb = KSimage.hsv2rgb(hsv)
    #                 KSimage.imwrite(rgb,savename)
    #
    #     duration = time.time() - t
    #     print('Finish preprocessing images from sample %d out of %d samples (%.2f sec)' % (i+1,len(log_file),duration))

    # jobs = []
    # for i, row in enumerate(log_file):
    #     p = multiprocessing.Process(target=para_preprocess, args=(row,i,log_file))
    #     jobs.append(p)
    #     p.start()
    #     # p.join()
    #
    # for j in jobs:
    #     j.join()

    data = list()
    for key in obj_list.keys():
        for i, row in enumerate(obj_list[key]):
            data.append((row, i, len(obj_list[key]), dict_path_preproc[key], dict_ext_preproc[key], dict_type[key]))

    mem = psutil.virtual_memory()
    npools = np.int(np.floor((mem.available / float(1024 ** 3)) / 4.4160))

    if npools > multiprocessing.cpu_count() - 2:
        npools = multiprocessing.cpu_count() - 2

    try:
        p = multiprocessing.Pool(npools)
        p.map(para_preprocess, data)
    finally:
        p.close()
        p.join()


######################################################################################
def para_register(args):
    row, i, len_log_file, dict_path_registered, dict_ext_registered = args
    t = time.time()

    img_name_dict = dict()
    for j, img_name in enumerate(row):
        img_name_dict[dict_path_registered.keys()[j]] = img_name

    img = dict()
    img['ck'] = KSimage.imread(img_name_dict['ck'])
    img['fish'] = KSimage.imread(img_name_dict['fish'])
    img['he'] = KSimage.imread(img_name_dict['he'])
    img['mask'] = KSimage.imread(img_name_dict['mask'])

    max0 = np.amax([img['ck'].shape[0], img['fish'].shape[0], img['he'].shape[0], img['mask'].shape[0]])
    max1 = np.amax([img['ck'].shape[1], img['fish'].shape[1], img['he'].shape[1], img['mask'].shape[0]])

    # make sure that the dimensions are equal between ck, fish, and he
    # first dim
    if img['ck'].shape[0] < max0:
        img['ck'] = np.pad(img['ck'], ((0, max0 - img['ck'].shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0)

    if img['fish'].shape[0] < max0:
        img['fish'] = np.pad(img['fish'], ((0, max0 - img['fish'].shape[0]), (0, 0), (0, 0)), 'constant',
                             constant_values=0)

    if img['he'].shape[0] < max0:
        img['he'] = np.pad(img['he'], ((0, max0 - img['he'].shape[0]), (0, 0), (0, 0)), 'constant', constant_values=255)

    if img['mask'].shape[0] < max0:
        img['mask'] = np.pad(img['mask'], ((0, max0 - img['mask'].shape[0]), (0, 0), (0, 0)), 'constant',
                             constant_values=0)

    # second dim
    if img['ck'].shape[1] < max1:
        img['ck'] = np.pad(img['ck'], ((0, 0), (0, max1 - img['ck'].shape[1]), (0, 0)), 'constant', constant_values=0)

    if img['fish'].shape[1] < max1:
        img['fish'] = np.pad(img['fish'], ((0, 0), (0, max1 - img['fish'].shape[1]), (0, 0)), 'constant',
                             constant_values=0)

    if img['he'].shape[1] < max1:
        img['he'] = np.pad(img['he'], ((0, 0), (0, max1 - img['he'].shape[1]), (0, 0)), 'constant', constant_values=255)

    if img['mask'].shape[1] < max1:
        img['mask'] = np.pad(img['mask'], ((0, 0), (0, max1 - img['mask'].shape[1]), (0, 0)), 'constant',
                             constant_values=0)

    # write fish
    basename = os.path.split(img_name_dict['fish'])[1]
    basename = os.path.splitext(basename)[0]
    basename = basename + dict_ext_registered['fish']
    savename = os.path.join(dict_path_registered['fish'], basename)

    KSimage.imwrite(img['fish'], savename)

    if ('ck' in img_name_dict.keys()) and ('fish' in img_name_dict.keys()):

        basename = os.path.split(img_name_dict['ck'])[1]
        basename = os.path.splitext(basename)[0]
        basename = basename + dict_ext_registered['ck']
        savename = os.path.join(dict_path_registered['ck'], basename)

        if not os.path.isfile(savename):
            # img['ck'] = KSimage.imread(img_name_dict['ck'])
            # img['fish'] = KSimage.imread(img_name_dict['fish'])
            registered_CK = KSimage.register_CK2FISH_translation(img['ck'], img['fish'])
            KSimage.imwrite(registered_CK, savename)

    if ('he' in img_name_dict.keys()) and ('fish' in img_name_dict.keys()):

        basename = os.path.split(img_name_dict['he'])[1]
        basename = os.path.splitext(basename)[0]
        basename = basename + dict_ext_registered['he']
        savename_he = os.path.join(dict_path_registered['he'], basename)

        basename = os.path.split(img_name_dict['mask'])[1]
        basename = os.path.splitext(basename)[0]
        basename = basename + dict_ext_registered['mask']
        savename_mask = os.path.join(dict_path_registered['mask'], basename)

        if not os.path.isfile(savename_he) or not os.path.isfile(savename_mask):
            # if not 'fish' in img.keys():
            #     img['fish'] = KSimage.imread(img_name_dict['fish'])
            # img['he'] = KSimage.imread(img_name_dict['he'])

            registered_HE, registered_mask = KSimage.register_HE2FISH_translation(img['he'], img['mask'], img['fish'])
            KSimage.imwrite(registered_HE, savename_he)
            KSimage.imwrite(registered_mask, savename_mask)

    duration = time.time() - t
    print('Finish registering images of sample %d out of %d samples (%.2f sec)' % (i + 1, len_log_file, duration))


###############################################
def register(dict_path_preproc, dict_ext_preproc,
             dict_path_registered, dict_ext_registered):
    obj_list = get_pair_list(dict_path_preproc, dict_ext_preproc)
    for key in dict_path_registered.keys():
        routine.create_dir(dict_path_registered[key])

    # for i, row in enumerate(log_file):
    #     t = time.time()
    #
    #     img_name_dict = dict()
    #     for j, img_name in enumerate(row):
    #         # img[dict_path_registered.keys()[j]] = KSimage.imread(img_name)
    #         img_name_dict[dict_path_registered.keys()[j]] = img_name
    #
    #     img = dict()
    #     if ('ck' in img_name_dict.keys()) and ('fish' in img_name_dict.keys()):
    #
    #         basename = os.path.split(img_name_dict['ck'])[1]
    #         basename = os.path.splitext(basename)[0]
    #         basename = basename + dict_ext_registered['ck']
    #         savename = os.path.join(dict_path_registered['ck'],basename)
    #
    #         if not os.path.isfile(savename):
    #             img['ck'] = KSimage.imread(img_name_dict['ck'])
    #             img['fish'] = KSimage.imread(img_name_dict['fish'])
    #             registered_CK = KSimage.register_CK2FISH_translation(img['ck'], img['fish'])
    #             KSregisterimage.imwrite(registered_CK,savename)
    #
    #     if ('he' in img_name_dict.keys()) and ('fish' in img_name_dict.keys()):
    #
    #         basename = os.path.split(img_name_dict['he'])[1]
    #         basename = os.path.splitext(basename)[0]
    #         basename = basename + dict_ext_registered['he']
    #         savename = os.path.join(dict_path_registered['he'], basename)
    #
    #         if not os.path.isfile(savename):
    #
    #             if not 'fish' in img.keys():
    #                 img['fish'] = KSimage.imread(img_name_dict['fish'])
    #             img['he'] = KSimage.imread(img_name_dict['he'])
    #
    #             registered_HE = KSimage.register_HE2FISH_translation(img['he'], img['fish'])
    #             KSimage.imwrite(registered_HE, savename)
    #
    #     duration = time.time() - t
    #     print('Finish registering images of sample %d out of %d samples (%.2f sec)' % (i+1,len(log_file),duration))

    data = list()
    len_log_file = len(obj_list[obj_list.keys()[0]])
    for i in range(len_log_file):
        row = list()
        for key in obj_list.keys():
            row.append(obj_list[key][i])
        data.append((row, i, len_log_file, dict_path_registered, dict_ext_registered))

    mem = psutil.virtual_memory()
    npools = np.int(np.floor((mem.available / float(1024 ** 3)) / 12.8))

    if npools > multiprocessing.cpu_count() - 2:
        npools = multiprocessing.cpu_count() - 2

    try:
        p = multiprocessing.Pool(npools)
        p.map(para_register, data)
    finally:
        p.close()
        p.join()


###############################################
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


###############################################
def he_dcis_segmentation(he_dir,
                         dict_ext,
                         he_dcis_segmentation_result_path,
                         gpu_list):

    routine.create_dir(he_dcis_segmentation_result_path)

    file_list = glob.glob(os.path.join(he_dir, '*' + dict_ext['he']))
    file_list = [[x] for x in file_list]

    from KS_lib.tf_model_he_dcis_segmentation import tf_model_main as main_he_dcis_segmentation
    from Modules.flags_he_dcis_segmentation import flags as flags_he_dcis_segmentation
    main_he_dcis_segmentation.main(1, 'test_model', flags_he_dcis_segmentation,
                                   file_list, he_dcis_segmentation_result_path,'0')

    #######################################
    # processes = set()
    # if gpu_list:
    #     max_processes = len(gpu_list)
    # else:
    #     max_processes = len(notebook_util.list_available_gpus())
    #     gpu_list = range(max_processes)
    #     gpu_list = [str(igpu) for igpu in gpu_list]
    #
    # split_he_log_file = chunkIt(he_log_file, max_processes)
    #
    # csv_file_list = list()
    # for i, item in enumerate(split_he_log_file):
    #     savename = os.path.join(he_dcis_segmentation_result_path, 'tmp_file_list' + str(i) + '.csv')
    #     csv_file_list.append(savename)
    #     KScsv.write_csv(item, savename)
    #
    # bash_script = os.path.join('Modules', 'he_dcis_segmentation.sh')
    # for igpu, csv_file in enumerate(csv_file_list):
    #     processes.add(subprocess.Popen([bash_script, csv_file, he_dcis_segmentation_result_path, gpu_list[igpu]]))
    #     if len(processes) >= max_processes:
    #         [p.wait() for p in processes]
    #         processes.difference_update([p for p in processes if p.poll() is not None])


###############################################
def para_ck_dcis_segmentation(args):
    i, he_file, ck_file, mask_file, ck_dcis_segmentation_result_path, len_log_file = args
    start_time = time.time()

    basename = os.path.split(ck_file)[1]
    basename = os.path.splitext(basename)[0]
    savename = os.path.join(ck_dcis_segmentation_result_path, basename + '.png')

    if not os.path.exists(savename):

        he = KSimage.imread(he_file)
        ck = KSimage.imread(ck_file)
        hand_draw_mask = KSimage.imread(mask_file)
        if hand_draw_mask.ndim == 3:
            hand_draw_mask = np.squeeze(hand_draw_mask, axis=2)

        # red channel from CK
        ck = ck / 255.0
        r = ck[:, :, 0] / (ck[:, :, 2] + np.spacing(1))
        epi = r > 1.5
        epi = KSimage.imfill(epi)
        epi = KSimage.bwareaopen(epi, 10 ** 2)
        epi = KSimage.imclose(epi, 10 ** 2)
        epi = KSimage.bwareaopen(epi, 50 ** 2)
        epi = KSimage.imdilate(epi, 50)

        # blood from HE
        he = he / 255.0
        r = he[:, :, 0] / (he[:, :, 2] + np.spacing(1))
        blood = r > 1.5
        blood = KSimage.imfill(blood)
        blood = KSimage.bwareaopen(blood, 10 ** 2)
        blood = KSimage.imclose(blood, 10 ** 2)
        blood = KSimage.bwareaopen(blood, 50 ** 2)
        blood = KSimage.imdilate(blood, 10 ** 2)

        mask = np.logical_and(epi, np.logical_not(blood))
        mask = KSimage.bwareaopen(mask, 300 ** 2)

        mask = mask * 255.0
        mask = mask.astype(np.uint8)

        mask[hand_draw_mask != 255.0] = 0

        KSimage.imwrite(mask, savename)

    duration = time.time() - start_time
    print('Finish segmenting DCIS regions from the CK image of sample %d out of %d samples (%.2f sec)' % (
    i + 1, len_log_file, duration))


###############################################
def ck_dcis_segmentation(dict_path_registered,
                         dict_ext_registered,
                         ck_dcis_segmentation_result_path):
    routine.create_dir(ck_dcis_segmentation_result_path)

    obj_list = get_pair_list(dict_path_registered, dict_ext_registered)

    # for i, row in enumerate(log_file):
    #     start_time = time.time()
    #
    #     he_file = row[idx_he]
    #     ck_file = row[idx_ck]
    #
    #     basename = os.path.split(ck_file)[1]
    #     basename = os.path.splitext(basename)[0]
    #     savename = os.path.join(ck_dcis_segmentation_result_path, basename + '.png')
    #
    #     if not os.path.exists(savename):
    #
    #         he = KSimage.imread(he_file)
    #         ck = KSimage.imread(ck_file)
    #
    #         # red channel from CK
    #         ck = ck/255.0
    #         r = ck[:,:,0]/(ck[:,:,2] + np.spacing(1))
    #         epi = r > 1.5
    #         epi = KSimage.imfill(epi)
    #         epi = KSimage.bwareaopen(epi,10**2)
    #         epi = KSimage.imclose(epi,10**2)
    #         epi = KSimage.bwareaopen(epi,50**2)
    #         epi = KSimage.imdilate(epi,50)
    #
    #         # blood from HE
    #         he = he/255.0
    #         r = he[:,:,0]/(he[:,:,2] + np.spacing(1))
    #         blood = r > 1.5
    #         blood = KSimage.imfill(blood)
    #         blood = KSimage.bwareaopen(blood,10**2)
    #         blood = KSimage.imclose(blood,10**2)
    #         blood = KSimage.bwareaopen(blood,50**2)
    #         blood = KSimage.imdilate(blood,10**2)
    #
    #         mask = np.logical_and(epi,np.logical_not(blood))
    #         mask = KSimage.bwareaopen(mask, 300**2)
    #
    #         mask = mask*255.0
    #         mask = mask.astype(np.uint8)
    #
    #         KSimage.imwrite(mask,savename)
    #
    #     duration = time.time() - start_time
    #     print('Finish segmenting DCIS regions from the CK image of sample %d out of %d samples (%.2f sec)' % (i + 1, len(log_file), duration))

    data = list()

    len_log_file = len(obj_list['he'])
    for i in range(len_log_file):
        he_file = obj_list['he'][i]
        ck_file = obj_list['ck'][i]
        mask_file = obj_list['mask'][i]
        data.append((i, he_file, ck_file, mask_file, ck_dcis_segmentation_result_path, len_log_file))

    mem = psutil.virtual_memory()
    npools = np.int(np.floor((mem.available / float(1024 ** 3)) / 4.2255))

    if npools > multiprocessing.cpu_count() - 2:
        npools = multiprocessing.cpu_count() - 2

    try:
        p = multiprocessing.Pool(npools)
        p.map(para_ck_dcis_segmentation, data)
    finally:
        p.close()
        p.join()


###############################################
def he_cell_segmentation(he_registered_path, dict_ext,
                         he_dcis_segmentation_result_path,
                         he_cell_segmentation_result_path,
                         gpu_list):
    routine.create_dir(he_cell_segmentation_result_path)

    file_list = glob.glob(os.path.join(he_registered_path, '*' + dict_ext['he']))
    file_list = [[x] for x in file_list]

    from KS_lib.tf_model_he_cell_segmentation import tf_model_main as main_he_cell_segmentation
    from Modules.flags_he_cell_segmentation import flags as flags_he_cell_segmentation
    main_he_cell_segmentation.main(1, 'test_model', flags_he_cell_segmentation,
                                   file_list, he_cell_segmentation_result_path,
                                   he_dcis_segmentation_result_path, str(0))

    # #######################################
    # processes = set()
    # if gpu_list:
    #     max_processes = len(gpu_list)
    # else:
    #     max_processes = len(notebook_util.list_available_gpus())
    #     gpu_list = range(max_processes)
    #     gpu_list = [str(igpu) for igpu in gpu_list]
    #
    # split_file = chunkIt(file_list, max_processes)
    #
    # csv_file_list = list()
    # for i, item in enumerate(split_file):
    #     savename = os.path.join(he_cell_segmentation_result_path, 'tmp_file_list' + str(i) + '.csv')
    #     csv_file_list.append(savename)
    #     KScsv.write_csv(item, savename)
    #
    # bash_script = os.path.join('Modules', 'he_cell_segmentation.sh')
    # for igpu, csv_file in enumerate(csv_file_list):
    #     processes.add(subprocess.Popen([bash_script, csv_file, he_cell_segmentation_result_path,
    #                                     he_dcis_segmentation_result_path, gpu_list[igpu]]))
    #     if len(processes) >= max_processes:
    #         [p.wait() for p in processes]
    #         processes.difference_update([p for p in processes if p.poll() is not None])
    #
    # #######################################


###############################################
def fish_cell_segmentation(fish_registered_path,
                           he_registered_path,
                           he_dcis_segmentation_result_path,
                           dcis_cell_segmentation_result_path,
                           gpu_list):
    routine.create_dir(dcis_cell_segmentation_result_path)

    dict_path = {'fish': fish_registered_path,
                 'he': he_registered_path}

    dict_ext = {'fish': '.png',
                'he': '.png'}

    obj_dict = get_pair_list(dict_path, dict_ext)
    nobjs = len(obj_dict[obj_dict.keys()[0]])

    file_list = list()
    for i in range(nobjs):
        x = list()
        for key in obj_dict.keys():
            x.append(obj_dict[key][i])
        file_list.append(x)

    # file_list = glob.glob(os.path.join(fish_registered_path,'*.png'))
    # file_list = [[x] for x in file_list]

    # from KS_lib.tf_model_dcis_cell_segmentation import tf_model_main as main_dcis_cell_segmentation
    # from Modules.flags_dcis_cell_segmentation import flags as flags_dcis_cell_segmentation
    # main_dcis_cell_segmentation.main(1, 'test_model', flags_dcis_cell_segmentation,
    #                                file_list, dcis_cell_segmentation_result_path,
    #                                he_dcis_segmentation_result_path,'0')

    #######################################
    processes = set()
    if gpu_list:
        max_processes = len(gpu_list)
    else:
        max_processes = len(notebook_util.list_available_gpus())
        gpu_list = range(max_processes)
        gpu_list = [str(igpu) for igpu in gpu_list]

    split_file = chunkIt(file_list, max_processes)

    csv_file_list = list()
    for i, item in enumerate(split_file):
        savename = os.path.join(dcis_cell_segmentation_result_path, 'tmp_file_list' + str(i) + '.csv')
        csv_file_list.append(savename)
        KScsv.write_csv(item, savename)

    bash_script = os.path.join('Modules', 'dcis_cell_segmentation.sh')
    for igpu, csv_file in enumerate(csv_file_list):
        processes.add(subprocess.Popen([bash_script, csv_file, dcis_cell_segmentation_result_path,
                                        he_dcis_segmentation_result_path, gpu_list[igpu]]))
        if len(processes) >= max_processes:
            [p.wait() for p in processes]
            processes.difference_update([p for p in processes if p.poll() is not None])

    #######################################


####################################################################################
def retouch_fish_cell_segmentation(fish_cell_seg, dcis_mask):
    if fish_cell_seg.ndim == 3:
        if fish_cell_seg.shape[2] == 1:
            fish_cell_seg = np.squeeze(fish_cell_seg, axis=2)

    if dcis_mask.ndim == 3:
        if dcis_mask.shape[2] == 1:
            dcis_mask = np.squeeze(dcis_mask, axis=2)

    bg = fish_cell_seg == 0
    bg = KSimage.bwareaopen(bg, 10 ** 2)
    bg = np.logical_not(bg)
    bg = (bg * 255.0).astype(np.uint8)
    _, contours, _ = cv2.findContours(bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bg_perim = np.zeros((bg.shape[0], bg.shape[1]), dtype=np.uint8)
    cv2.drawContours(bg_perim, contours, -1, (255), 3)
    bg_perim = bg_perim == 255

    inner = fish_cell_seg == 1
    outer = fish_cell_seg == 2
    outer = np.logical_not(KSimage.bwareaopen(np.logical_not(outer), 10 ** 2))

    cells_skel = KSimage.skeleton(np.logical_or(outer, bg_perim))
    cells_skel = KSimage.imclose(cells_skel, 10)

    cells_fill_holes = KSimage.imfill(cells_skel)
    cells_fill_holes = KSimage.imopen(cells_fill_holes, 2)
    cells_fill_holes = KSimage.bwareaopen(cells_fill_holes, 10 ** 2)

    difference = np.logical_and(cells_fill_holes, np.logical_not(cells_skel))
    difference = KSimage.bwareaopen(difference, 10 ** 2)
    difference = KSimage.imerode(difference, 1)

    L, _ = KSimage.bwlabel(difference)
    idx = KSimage.label2idx(L)
    del idx[0]

    for key in idx.keys():
        idx_pair = np.unravel_index(idx[key], difference.shape)
        if np.sum(inner[idx_pair]) / np.float(len(idx[key])) > 0.1:
            difference[idx_pair] = False

    difference = KSimage.imclose(difference, 2)

    retouch_cells = np.logical_and(cells_fill_holes, np.logical_not(difference))
    retouch_cells = KSimage.imopen(retouch_cells, 3)
    retouch_cells = KSimage.bwareaopen(retouch_cells, 25 ** 2)

    boundary = KSimage.imdilate(cells_skel, 2)
    canvas = np.zeros(retouch_cells.shape, np.uint8)
    canvas[retouch_cells] = 1
    canvas[boundary] = 2

    # remove anything outside the mask
    dcis_mask = dcis_mask == 255.0
    canvas[dcis_mask == 0.0] = 0

    return canvas


####################################################################################
def remove_he_cell_seg(retouch_cell_seg, he_cell_seg):
    retouch_cell_seg = retouch_cell_seg == 1

    he_cell_seg = he_cell_seg.astype(np.bool)
    he_cell_seg = KSimage.bwareaopen(he_cell_seg, 10 ** 2)
    L, _ = KSimage.bwlabel(he_cell_seg)

    idx = KSimage.label2idx(L)
    del idx[0]

    for key in idx.keys():
        idx_pair = np.unravel_index(idx[key], retouch_cell_seg.shape)
        ratio = np.sum(retouch_cell_seg[idx_pair]) / np.float32(idx[key].size)
        if ratio < 0.5:
            he_cell_seg[idx_pair] = False

    he_cell_seg = KSimage.bwareaopen(he_cell_seg, 10 ** 2)

    return he_cell_seg


####################################################################################
def separate_fish_cells(retouch_fish_cell_seg, he_cell_seg, fish, fish_cell_seg):
    # dapi = fish[:, :, 2]
    dapi = fish_cell_seg[:, :, 1]

    if retouch_fish_cell_seg.dtype == 'bool':
        retouch_cell_seg = 255 * retouch_fish_cell_seg
        retouch_cell_seg = retouch_cell_seg.astype(np.float32)

    if he_cell_seg.astype != 'bool':
        he_cell_seg = he_cell_seg.astype(np.bool)

    sx = ndimage.sobel(retouch_cell_seg, axis=0, mode='reflect')
    sy = ndimage.sobel(retouch_cell_seg, axis=1, mode='reflect')
    gradmag = np.sqrt(sx ** 2 + sy ** 2)
    # dist = (255.0 - KSimage.gaussian_filter(dapi, 5)) + gradmag
    dist = (255.0 - dapi) + gradmag
    markers, _ = KSimage.bwlabel(he_cell_seg)

    if markers.ndim == 3:
        if markers.shape[2] == 1:
            markers = np.squeeze(markers, axis=2)

    if dist.ndim == 3:
        if dist.shape[2] == 1:
            dist = np.squeeze(dist, axis=2)

    if retouch_fish_cell_seg.ndim == 3:
        if retouch_fish_cell_seg.shape[2] == 1:
            retouch_fish_cell_seg = np.squeeze(retouch_fish_cell_seg, axis=2)

    separated_cell_mask = watershed(dist, markers, mask=retouch_fish_cell_seg)

    return separated_cell_mask


####################################################################################
def classify_and_remove_cells(ck, he, he_cell_seg, retouch_fish_cells):
    separated_cell_mask, _ = KSimage.bwlabel(retouch_fish_cells == 1)

    if he_cell_seg.dtype != 'bool':
        he_cell_seg = he_cell_seg.astype(np.bool)

    if he_cell_seg.ndim == 3:
        if he_cell_seg.shape[2] == 1:
            he_cell_seg = np.squeeze(he_cell_seg, axis=2)

    if separated_cell_mask.ndim == 3:
        if separated_cell_mask.shape[2] == 1:
            separated_cell_mask = np.squeeze(separated_cell_mask, axis=2)

    ck = ck.astype(np.float32) / 255.0
    ratio = ck[:, :, 0] / (ck[:, :, 2] + np.spacing(1))
    bw = ratio > 1.5
    bw = KSimage.imfill(bw)
    # ck_mask = KSimage.bwareaopen(bw, 10**2)
    ck_mask = KSimage.bwareaopen(bw, 30 ** 2)

    # blood mask
    he = he.astype(np.float32) / 255.0
    ratio = he[:, :, 0] / (he[:, :, 2] + np.spacing(1))
    blood = ratio > 1.5
    blood = KSimage.imfill(blood)
    blood_mask = KSimage.bwareaopen(blood, 10 ** 2)

    he_cell_seg_uint8 = 255 * (he_cell_seg > 0)
    he_cell_seg_uint8 = he_cell_seg_uint8.astype(np.uint8)

    prop = cv2.connectedComponentsWithStats(he_cell_seg_uint8, connectivity=8, ltype=cv2.CV_32S)
    centroid = prop[3]
    centroid = np.delete(centroid, 0, axis=0)
    centroid = centroid.astype(np.uint16)

    he_cell_seg_label = prop[1]
    he_idx = KSimage.label2idx(he_cell_seg_label)
    del he_idx[0]

    L = np.copy(separated_cell_mask)
    idx = KSimage.label2idx(L)
    del idx[0]

    ncells_map = np.zeros(L.shape, dtype=np.uint16)

    for key in idx.keys():
        idx_pair = np.unravel_index(idx[key], L.shape)

        # check if the segment is too big than the total sizes of cells
        if np.float(idx[key].size) / np.float(np.sum(he_cell_seg[idx_pair]) + np.spacing(1)) > 10.0:
            L[idx_pair] = 0
            ncells_map[idx_pair] = 0

        else:
            # if the size is okay next check how many cells are epi
            x = he_cell_seg_label[idx_pair]
            index = np.argwhere(x == 0)
            x = np.delete(x, index)

            if x.size:
                list_cells_to_check = list()
                candidate = np.unique(x)
                for cell in candidate:
                    intersect_list = np.intersect1d(he_idx[cell], idx[key])
                    if np.float(intersect_list.size) / np.float(he_idx[cell].size) > 0.5:
                        list_cells_to_check.append(cell)

                # if no cell in the check list continue to the next loop
                if len(list_cells_to_check) == 0:
                    L[idx_pair] = 0
                    ncells_map[idx_pair] = 0
                    continue

                try:
                    ncells_map[idx_pair] = len(list_cells_to_check)
                except:
                    ncells_map[idx_pair] = 1
                    list_cells_to_check = [list_cells_to_check]

                cell_flags = np.zeros(len(list_cells_to_check), dtype=bool)
                for icell, cell in enumerate(list_cells_to_check):
                    center = centroid[cell - 1, :]

                    try:
                        expand = 40
                        ext_ck = ck_mask[center[1] - expand:center[1] + expand, center[0] - expand:center[0] + expand]
                        ext_blood = blood_mask[center[1] - expand:center[1] + expand,
                                    center[0] - expand:center[0] + expand]

                        if (not np.any(ext_ck)) or (np.any(ext_blood)):
                            # if (not np.any(ext_ck)):
                            cell_flags[icell] = False
                        else:
                            cell_flags[icell] = True
                    except:
                        cell_flags[icell] = False

                if np.sum(cell_flags) / np.float(len(list_cells_to_check)) < 0.5:
                    L[idx_pair] = 0
                    ncells_map[idx_pair] = 0

            else:
                L[idx_pair] = 0
                ncells_map[idx_pair] = 0

    return L, ncells_map


####################################################################################
def para_cell_sep(args):
    i, obj_dict,
    retouched_he_cell_seg_result_path,
    retouched_fish_cell_seg_result_path,
    cell_separation_result_path, ncells_result_path, nobjs = args
    start_time = time.time()
    basename = os.path.split(obj_dict['fish_cell_seg'][i])[1]
    basename = os.path.splitext(basename)[0]
    basename = basename + '.png'
    savename_fish_seg = os.path.join(retouched_fish_cell_seg_result_path, basename)
    savename1 = os.path.join(cell_separation_result_path, basename)
    savename2 = os.path.join(ncells_result_path, basename)

    basename = os.path.split(obj_dict['he_cell_seg'][i])[1]
    basename = os.path.splitext(basename)[0]
    basename = basename + '.png'
    savename_he_seg = os.path.join(retouched_he_cell_seg_result_path, basename)

    if not os.path.isfile(savename1) or not os.path.isfile(savename2) or not os.path.isfile(
            savename_fish_seg) or not os.path.isfile(savename_he_seg):
        he = KSimage.imread(obj_dict['he'][i])
        # fish = KSimage.imread(obj_dict['fish'][i])
        ck = KSimage.imread(obj_dict['ck'][i])
        he_cell_seg = KSimage.imread(obj_dict['he_cell_seg'][i])
        fish_cell_seg = KSimage.imread(obj_dict['fish_cell_seg'][i])
        dcis_mask = KSimage.imread(obj_dict['dcis_mask'][i])

        retouched_fish_cells = retouch_fish_cell_segmentation(fish_cell_seg, dcis_mask)
        retouched_he_cells = remove_he_cell_seg(retouched_fish_cells, he_cell_seg)
        # separated_cell_mask = separate_fish_cells(retouched_fish_cells, he_cell_seg, fish, fish_cell_seg)
        epithelial_cell_mask, ncells_map = classify_and_remove_cells(ck, he, retouched_he_cells, retouched_fish_cells)

        if retouched_he_cells.dtype == 'bool':
            retouched_he_cells = (retouched_he_cells * 255.0).astype(np.uint8)

        if retouched_fish_cells.dtype == 'bool':
            retouched_fish_cells = (retouched_fish_cells * 255.0).astype(np.uint8)

        KSimage.imwrite(retouched_fish_cells, savename_fish_seg)
        KSimage.imwrite(retouched_he_cells, savename_he_seg)
        KSimage.imwrite(epithelial_cell_mask.astype(np.uint16), savename1)
        KSimage.imwrite(ncells_map.astype(np.uint16), savename2)

    duration = time.time() - start_time
    print('Finish separating cells in %d sample out of %d images (%.2f)' % (i + 1, nobjs, duration))


####################################################################################
def cell_separation(registered_he_path, registered_fish_path, registered_ck_path,
                    he_cell_seg_result_path, fish_cell_seg_result_path,
                    dcis_mask_path,
                    retouched_he_cell_seg_result_path,
                    retouched_fish_cell_seg_result_path,
                    cell_separation_result_path,
                    ncells_result_path):
    dict_path = {'he': registered_he_path,
                 # 'fish':registered_fish_path,
                 'ck': registered_ck_path,
                 'he_cell_seg': he_cell_seg_result_path,
                 'fish_cell_seg': fish_cell_seg_result_path,
                 'dcis_mask': dcis_mask_path}

    dict_ext = {'he': '.png',
                # 'fish':'.png',
                'ck': '.png',
                'he_cell_seg': '.png',
                'fish_cell_seg': '.png',
                'dcis_mask': '.png'}

    obj_dict = get_pair_list(dict_path, dict_ext)

    nobjs = len(obj_dict[obj_dict.keys()[0]])

    routine.create_dir(retouched_he_cell_seg_result_path)
    routine.create_dir(retouched_fish_cell_seg_result_path)
    routine.create_dir(cell_separation_result_path)
    routine.create_dir(ncells_result_path)

    # for i in range(nobjs):
    #     start_time = time.time()
    #     basename = os.path.split(obj_dict['fish_cell_seg'][i])[1]
    #     basename = os.path.splitext(basename)[0]
    #     basename = basename + '.png'
    #     savename_fish_seg = os.path.join(retouched_fish_cell_seg_result_path,basename)
    #     savename1 = os.path.join(cell_separation_result_path, basename)
    #     savename2 = os.path.join(ncells_result_path, basename)
    #
    #     basename = os.path.split(obj_dict['he_cell_seg'][i])[1]
    #     basename = os.path.splitext(basename)[0]
    #     basename = basename + '.png'
    #     savename_he_seg = os.path.join(retouched_he_cell_seg_result_path,basename)
    #
    #     if not os.path.isfile(savename1) or not os.path.isfile(savename2) or not os.path.isfile(savename_fish_seg) or not os.path.isfile(savename_he_seg):
    #         he = KSimage.imread(obj_dict['he'][i])
    #         # fish = KSimage.imread(obj_dict['fish'][i])
    #         ck = KSimage.imread(obj_dict['ck'][i])
    #         he_cell_seg = KSimage.imread(obj_dict['he_cell_seg'][i])
    #         fish_cell_seg = KSimage.imread(obj_dict['fish_cell_seg'][i])
    #
    #         retouched_fish_cells = retouch_fish_cell_segmentation(fish_cell_seg)
    #         retouched_he_cells = remove_he_cell_seg(retouched_fish_cells, he_cell_seg)
    #         # separated_cell_mask = separate_fish_cells(retouched_fish_cells, he_cell_seg, fish, fish_cell_seg)
    #         epithelial_cell_mask, ncells_map = classify_and_remove_cells(ck, he, retouched_he_cells, retouched_fish_cells)
    #
    #         if retouched_he_cells.dtype == 'bool':
    #             retouched_he_cells = (retouched_he_cells*255.0).astype(np.uint8)
    #
    #         if retouched_fish_cells.dtype == 'bool':
    #             retouched_fish_cells = (retouched_fish_cells*255.0).astype(np.uint8)
    #
    #         KSimage.imwrite(retouched_fish_cells,savename_fish_seg)
    #         KSimage.imwrite(retouched_he_cells,savename_he_seg)
    #         KSimage.imwrite(epithelial_cell_mask.astype(np.uint16),savename1)
    #         KSimage.imwrite(ncells_map.astype(np.uint16),savename2)
    #
    #     duration = time.time() - start_time
    #     print('Finish separating cells in %d sample out of %d images (%.2f)' % (i+1,nobjs,duration))

    data = list()
    for i in range(nobjs):
        data.append((i, obj_dict, retouched_he_cell_seg_result_path, retouched_fish_cell_seg_result_path,
                     cell_separation_result_path, ncells_result_path, nobjs))

    mem = psutil.virtual_memory()
    npools = np.int(np.floor((mem.available / float(1024 ** 3)) / 10.0))

    if npools > multiprocessing.cpu_count() - 2:
        npools = multiprocessing.cpu_count() - 2

    try:
        p = multiprocessing.Pool(npools)
        p.map(para_cell_sep, data)
    finally:
        p.close()
        p.join()


####################################################################################
def dcis_signal_detection_green(fish_registered_path,
                                he_dcis_segmentation_result_path,
                                result_path,
                                gpu_list):
    file_list = glob.glob(os.path.join(fish_registered_path, '*.png'))
    file_list = [[x] for x in file_list]

    routine.create_dir(result_path)

    # from KS_lib.tf_model_probe_detection import tf_model_main as main_probe_detection
    # from Modules.flags_probe_detection_green import flags as flags_probe_detection_green
    # main_probe_detection.main(1, 'test_model', flags_probe_detection_green,
    #                                  file_list, result_path,
    #                                  he_dcis_segmentation_result_path, '0')

    #######################################
    processes = set()
    if gpu_list:
        max_processes = len(gpu_list)
    else:
        max_processes = len(notebook_util.list_available_gpus())
        gpu_list = range(max_processes)
        gpu_list = [str(igpu) for igpu in gpu_list]

    split_file = chunkIt(file_list, max_processes)

    csv_file_list = list()
    for i, item in enumerate(split_file):
        savename = os.path.join(result_path, 'tmp_file_list' + str(i) + '.csv')
        csv_file_list.append(savename)
        KScsv.write_csv(item, savename)

    bash_script = os.path.join('Modules', 'probe_detection_green.sh')
    for igpu, csv_file in enumerate(csv_file_list):
        processes.add(subprocess.Popen([bash_script, csv_file, result_path,
                                        he_dcis_segmentation_result_path, gpu_list[igpu]]))
        if len(processes) >= max_processes:
            [p.wait() for p in processes]
            processes.difference_update([p for p in processes if p.poll() is not None])

            #######################################


####################################################################################
def dcis_signal_detection_red(fish_registered_path,
                              he_dcis_segmentation_result_path,
                              result_path,
                              gpu_list):
    file_list = glob.glob(os.path.join(fish_registered_path, '*.png'))
    file_list = [[x] for x in file_list]

    routine.create_dir(result_path)

    # from KS_lib.tf_model_probe_detection import tf_model_main as main_probe_detection
    # from Modules.flags_probe_detection_red import flags as flags_probe_detection_red
    # main_probe_detection.main(1, 'test_model', flags_probe_detection_red,
    #                                  file_list, result_path,
    #                                  he_dcis_segmentation_result_path, '0')

    #######################################
    processes = set()
    if gpu_list:
        max_processes = len(gpu_list)
    else:
        max_processes = len(notebook_util.list_available_gpus())
        gpu_list = range(max_processes)
        gpu_list = [str(igpu) for igpu in gpu_list]

    split_file = chunkIt(file_list, max_processes)

    csv_file_list = list()
    for i, item in enumerate(split_file):
        savename = os.path.join(result_path, 'tmp_file_list' + str(i) + '.csv')
        csv_file_list.append(savename)
        KScsv.write_csv(item, savename)

    bash_script = os.path.join('Modules', 'probe_detection_red.sh')
    for igpu, csv_file in enumerate(csv_file_list):
        processes.add(subprocess.Popen([bash_script, csv_file, result_path,
                                        he_dcis_segmentation_result_path, gpu_list[igpu]]))
        if len(processes) >= max_processes:
            [p.wait() for p in processes]
            processes.difference_update([p for p in processes if p.poll() is not None])

            #######################################


####################################################################################
def dcis_signal_detection_yellow(fish_registered_path,
                              he_dcis_segmentation_result_path,
                              result_path,
                              gpu_list):
    file_list = glob.glob(os.path.join(fish_registered_path, '*.png'))
    file_list = [[x] for x in file_list]

    routine.create_dir(result_path)

    # from KS_lib.tf_model_probe_detection import tf_model_main as main_probe_detection
    # from Modules.flags_probe_detection_yellow import flags as flags_probe_detection_yellow
    # main_probe_detection.main(1, 'test_model', flags_probe_detection_yellow,
    #                                  file_list, result_path,
    #                                  he_dcis_segmentation_result_path, '0')

    #######################################
    processes = set()
    if gpu_list:
        max_processes = len(gpu_list)
    else:
        max_processes = len(notebook_util.list_available_gpus())
        gpu_list = range(max_processes)
        gpu_list = [str(igpu) for igpu in gpu_list]

    split_file = chunkIt(file_list, max_processes)

    csv_file_list = list()
    for i, item in enumerate(split_file):
        savename = os.path.join(result_path, 'tmp_file_list' + str(i) + '.csv')
        csv_file_list.append(savename)
        KScsv.write_csv(item, savename)

    bash_script = os.path.join('Modules', 'probe_detection_yellow.sh')
    for igpu, csv_file in enumerate(csv_file_list):
        processes.add(subprocess.Popen([bash_script, csv_file, result_path,
                                        he_dcis_segmentation_result_path, gpu_list[igpu]]))
        if len(processes) >= max_processes:
            [p.wait() for p in processes]
            processes.difference_update([p for p in processes if p.poll() is not None])

            #######################################


####################################################################################
def para_generate_coordinate_green(args):
    file, i, result_path, len_file_list = args
    start_time = time.time()

    basename = os.path.split(file)[1]
    basename = os.path.splitext(basename)[0]
    savename = os.path.join(result_path, basename + '.mat')

    if not os.path.isfile(savename):
        img = KSimage.imread(file)
        coordinate = np.array(KSimage.find_local_maxima(np.float32(img), min_distance=4, threshold_rel=0.52))
        #####################################################################
        # adding more coordinate using an area criterion to compensate with the detection resolution
        mask = img > np.max(img) * 0.1
        L, _ = KSimage.bwlabel(mask)
        idx = KSimage.label2idx(L)
        del idx[0]

        coord_mask = np.zeros(mask.shape, np.uint8)
        for row in coordinate:
            coord_mask[row[0], row[1]] = 1

        for key in idx.keys():
            coord = np.unravel_index(idx[key], mask.shape)
            n_detect = np.sum(coord_mask[coord])
            if n_detect:
                n_adjusted = np.round(idx[key].size / float(np.pi * (4.9 ** 2)))
                n_diff = n_adjusted - n_detect

                for n in range(int(n_diff)):
                    coord_yes = coord[0][coord_mask[coord] == 1], coord[1][coord_mask[coord] == 1]
                    coord_no = coord[0][coord_mask[coord] == 0], coord[1][coord_mask[coord] == 0]

                    dist = cdist(np.vstack(coord_no).transpose(), np.vstack(coord_yes).transpose(), 'euclidean')
                    max_idx = np.argmax(np.sum(dist, axis=1))
                    coord_mask[coord_no[0][max_idx], coord_no[1][max_idx]] = 1
                    coordinate = np.vstack((coordinate, [coord_no[0][max_idx], coord_no[1][max_idx]]))

            # print('object %d of %d' % (key, len(idx.keys())))
        ####################################################################
        coordinate = np.round(coordinate / 2.0).astype(np.int)

        matlab.save(savename, {'coordinate': coordinate})

    duration = time.time() - start_time
    print('Finish extracting coordinates of green signals from sample %d out of %d samples (%.2f)' %
          (i + 1, len_file_list, duration))


####################################################################################
def para_generate_coordinate_red(args):
    file, i, result_path, len_file_list = args
    start_time = time.time()

    basename = os.path.split(file)[1]
    basename = os.path.splitext(basename)[0]
    savename = os.path.join(result_path, basename + '.mat')

    if not os.path.isfile(savename):
        img = KSimage.imread(file)
        coordinate = np.array(KSimage.find_local_maxima(np.float32(img), min_distance=4, threshold_rel=0.53))
        #####################################################################
        # adding more coordinate using an area criterion to compensate with the detection resolution
        mask = img > np.max(img) * 0.1
        L, _ = KSimage.bwlabel(mask)
        idx = KSimage.label2idx(L)
        del idx[0]

        coord_mask = np.zeros(mask.shape, np.uint8)
        for row in coordinate:
            coord_mask[row[0], row[1]] = 1

        for key in idx.keys():
            coord = np.unravel_index(idx[key], mask.shape)
            n_detect = np.sum(coord_mask[coord])
            if n_detect:
                n_adjusted = np.round(idx[key].size / float(np.pi * (5.0 ** 2)))
                n_diff = n_adjusted - n_detect

                # erode_coord = np.zeros(mask.shape, np.uint8)
                # erode_coord[coord] = 1
                # erode_coord = np.expand_dims(KSimage.imerode(erode_coord, 5), axis=2)
                # erode_coord = np.where(erode_coord)
                #
                # coord_no = erode_coord[0], erode_coord[1]
                for n in range(int(n_diff)):
                    coord_yes = coord[0][coord_mask[coord] == 1], coord[1][coord_mask[coord] == 1]
                    coord_no = coord[0][coord_mask[coord] == 0], coord[1][coord_mask[coord] == 0]

                    dist = cdist(np.vstack(coord_no).transpose(), np.vstack(coord_yes).transpose(), 'euclidean')
                    max_idx = np.argmax(np.sum(dist, axis=1))
                    coord_mask[coord_no[0][max_idx], coord_no[1][max_idx]] = 1
                    coordinate = np.vstack((coordinate, [coord_no[0][max_idx], coord_no[1][max_idx]]))

            # print('object %d of %d' % (key, len(idx.keys())))
        ####################################################################
        coordinate = np.round(coordinate / 2.0).astype(np.int)

        matlab.save(savename, {'coordinate': coordinate})

    duration = time.time() - start_time
    print('Finish extracting coordinates of red signals from sample %d out of %d samples (%.2f)' %
          (i + 1, len_file_list, duration))


####################################################################################
def para_generate_coordinate_yellow(args):
    (file, i, result_path, len_file_list) = args
    start_time = time.time()

    basename = os.path.split(file)[1]
    basename = os.path.splitext(basename)[0]
    savename = os.path.join(result_path, basename + '.mat')

    if not os.path.isfile(savename):
        img = KSimage.imread(file)
        coordinate = np.array(KSimage.find_local_maxima(np.float32(img), min_distance=4, threshold_rel=0.63))
        #####################################################################
        # adding more coordinate using an area criterion to compensate with the detection resolution
        mask = img > np.max(img) * 0.1
        L, _ = KSimage.bwlabel(mask)
        idx = KSimage.label2idx(L)
        del idx[0]

        coord_mask = np.zeros(mask.shape, np.uint8)
        for row in coordinate:
            coord_mask[row[0], row[1]] = 1

        for key in idx.keys():
            coord = np.unravel_index(idx[key], mask.shape)
            n_detect = np.sum(coord_mask[coord])
            if n_detect:
                n_adjusted = np.round(idx[key].size / float(np.pi * (5.0 ** 2)))
                n_diff = n_adjusted - n_detect

                # erode_coord = np.zeros(mask.shape, np.uint8)
                # erode_coord[coord] = 1
                # erode_coord = np.expand_dims(KSimage.imerode(erode_coord, 5), axis=2)
                # erode_coord = np.where(erode_coord)
                #
                # coord_no = erode_coord[0], erode_coord[1]
                for n in range(int(n_diff)):
                    coord_yes = coord[0][coord_mask[coord] == 1], coord[1][coord_mask[coord] == 1]
                    coord_no = coord[0][coord_mask[coord] == 0], coord[1][coord_mask[coord] == 0]

                    dist = cdist(np.vstack(coord_no).transpose(), np.vstack(coord_yes).transpose(), 'euclidean')
                    max_idx = np.argmax(np.sum(dist, axis=1))
                    coord_mask[coord_no[0][max_idx], coord_no[1][max_idx]] = 1
                    coordinate = np.vstack((coordinate, [coord_no[0][max_idx], coord_no[1][max_idx]]))

            # print('object %d of %d' % (key, len(idx.keys())))
        ####################################################################
        coordinate = np.round(coordinate / 2.0).astype(np.int)

        matlab.save(savename, {'coordinate': coordinate})

    duration = time.time() - start_time
    print('Finish extracting coordinates of red signals from sample %d out of %d samples (%.2f)' %
          (i + 1, len_file_list, duration))


####################################################################################
def generate_coordinate_green(detection_path, result_path):
    routine.create_dir(result_path)

    file_list = glob.glob(os.path.join(detection_path, '*.png'))

    # for i, file in enumerate(file_list):
    #     start_time = time.time()
    #
    #     basename = os.path.split(file)[1]
    #     basename = os.path.splitext(basename)[0]
    #     savename = os.path.join(result_path,basename+'.mat')
    #
    #     if not os.path.isfile(savename):
    #         img = KSimage.imread(file)
    #         coordinate = np.array(KSimage.find_local_maxima(np.float32(img), min_distance=4, threshold_rel=0.5))
    #         coordinate = np.round(coordinate / 2.0).astype(np.int)
    #
    #         matlab.save(savename,{'coordinate':coordinate})
    #
    #     duration = time.time() - start_time
    #     print('Finish extracting coordinates of green/red signals from sample %d out of %d samples (%.2f)' %
    #           (i+1, len(file_list), duration))

    data = list()
    for i, file in enumerate(file_list):
        data.append((file, i, result_path, len(file_list)))

    for dat in data:
        para_generate_coordinate_green(dat)

    mem = psutil.virtual_memory()
    npools = np.int(np.floor((mem.available / float(1024 ** 3)) / 3.0048))

    if npools > multiprocessing.cpu_count() - 2:
        npools = multiprocessing.cpu_count() - 2

    try:
        p = multiprocessing.Pool(npools)
        p.map(para_generate_coordinate_green, data)
    finally:
        p.close()
        p.join()


####################################################################################
def generate_coordinate_red(detection_path, result_path):
    routine.create_dir(result_path)

    file_list = glob.glob(os.path.join(detection_path, '*.png'))

    data = list()
    for i, file in enumerate(file_list):
        data.append((file, i, result_path, len(file_list)))

    mem = psutil.virtual_memory()
    npools = np.int(np.floor((mem.available / float(1024 ** 3)) / 3.0048))

    if npools > multiprocessing.cpu_count() - 2:
        npools = multiprocessing.cpu_count() - 2

    try:
        p = multiprocessing.Pool(npools)
        p.map(para_generate_coordinate_red, data)
    finally:
        p.close()
        p.join()


####################################################################################
def generate_coordinate_yellow(detection_path, result_path):
    routine.create_dir(result_path)

    file_list = glob.glob(os.path.join(detection_path, '*.png'))

    data = list()
    for i, file in enumerate(file_list):
        data.append((file, i, result_path, len(file_list)))

    mem = psutil.virtual_memory()
    npools = np.int(np.floor((mem.available / float(1024 ** 3)) / 3.0048))

    if npools > multiprocessing.cpu_count() - 2:
        npools = multiprocessing.cpu_count() - 2

    try:
        p = multiprocessing.Pool(npools)
        p.map(para_generate_coordinate_yellow, data)
    finally:
        p.close()
        p.join()


#######################################################################################
def para_count_signal(args):
    (iobj, dict_obj, result_path, nobj) = args
    start_time = time.time()

    basename = os.path.split(dict_obj['cell'][iobj])[1]
    basename = os.path.splitext(basename)[0]
    basename = basename + '.mat'
    savename = os.path.join(result_path, basename)

    if not os.path.isfile(savename):

        label_map = KSimage.imread(dict_obj['cell'][iobj])
        ncell_map = KSimage.imread(dict_obj['ncell'][iobj])

        if label_map.ndim == 3:
            if label_map.shape[2] == 1:
                label_map = np.squeeze(label_map, axis=2)

        coordinate_dict = dict()
        coordinate_dict['green'] = matlab.load(dict_obj['green'][iobj])['coordinate']
        coordinate_dict['red'] = matlab.load(dict_obj['red'][iobj])['coordinate']
        coordinate_dict['yellow'] = matlab.load(dict_obj['yellow'][iobj])['coordinate']

        canvas = dict()

        for key in coordinate_dict.keys():
            canvas[key] = np.zeros((label_map.shape[0], label_map.shape[1])).astype(np.bool)
            coordinate = coordinate_dict[key]
            for j in range(coordinate.shape[0]):
                canvas[key][coordinate[j, 0], coordinate[j, 1]] = True

        output = dict()
        # output['label_map'] = label_map
        output['label'] = list()
        output['coordinate'] = list()
        output['ncell'] = list()
        output['size'] = list()
        for key in coordinate_dict.keys():
            output[key] = list()

        idx = KSimage.label2idx(label_map)
        del idx[0]

        for i, cell_idx in enumerate(idx.keys()):
            idx_pair = np.unravel_index(idx[cell_idx], label_map.shape)
            output['label'].append(cell_idx)
            output['coordinate'].append(np.mean(idx_pair, axis=1))

            ncell = stats.mode(ncell_map[idx_pair])[0]
            output['ncell'].append(ncell)

            output['size'].append(idx[cell_idx].size)

            for key in coordinate_dict.keys():
                signal = canvas[key][idx_pair]
                output[key].append(np.sum(signal))

        if output['label']:
            output['label'] = np.vstack(output['label'])
            output['coordinate'] = np.vstack(output['coordinate'])
            output['ncell'] = np.vstack(output['ncell'])
            output['size'] = np.vstack(output['size'])
            output['green'] = np.vstack(output['green'])
            output['red'] = np.vstack(output['red'])
            output['yellow'] = np.vstack(output['yellow'])

        else:
            output['label'] = []
            output['coordinate'] = []
            output['ncell'] = []
            output['size'] = []
            output['green'] = []
            output['red'] = []
            output['yellow'] = []

        matlab.save(savename, {'output': output})

    duration = time.time() - start_time
    print('Finish counting signals from sample %d of %d samples (%.2f)' % (iobj + 1, nobj, duration))


#######################################################################################
def count_signal(cell_label_path, ncell_path, green_coordinate, red_coordinate, yellow_coordinate, result_path):
    routine.create_dir(result_path)

    dict_path = {'cell': cell_label_path,
                 'ncell': ncell_path,
                 'green': green_coordinate,
                 'red': red_coordinate,
                 'yellow': yellow_coordinate}

    dict_ext = {'cell': '.png',
                'ncell': '.png',
                'green': '.mat',
                'red': '.mat',
                'yellow': '.mat'}

    dict_obj = get_pair_list(dict_path, dict_ext)

    nobj = len(dict_obj[dict_obj.keys()[0]])

    # for iobj in range(nobj):
    #     start_time = time.time()
    #
    #     basename = os.path.split(dict_obj['cell'][iobj])[1]
    #     basename = os.path.splitext(basename)[0]
    #     basename = basename + '.mat'
    #     savename = os.path.join(result_path,basename)
    #
    #     if not os.path.isfile(savename):
    #
    #         label_map = KSimage.imread(dict_obj['cell'][iobj])
    #         ncell_map = KSimage.imread(dict_obj['ncell'][iobj])
    #
    #         if label_map.ndim == 3:
    #             if label_map.shape[2] == 1:
    #                 label_map = np.squeeze(label_map,axis = 2)
    #
    #         coordinate_dict = dict()
    #         coordinate_dict['green'] = matlab.load(dict_obj['green'][iobj])['coordinate']
    #         coordinate_dict['red'] = matlab.load(dict_obj['red'][iobj])['coordinate']
    #
    #         canvas = dict()
    #
    #         for key in coordinate_dict.keys():
    #             canvas[key] = np.zeros((label_map.shape[0],label_map.shape[1])).astype(np.bool)
    #             coordinate = coordinate_dict[key]
    #             for j in range(coordinate.shape[0]):
    #                 canvas[key][coordinate[j,0],coordinate[j,1]] = True
    #
    #
    #         output = dict()
    #         # output['label_map'] = label_map
    #         output['label'] = list()
    #         output['coordinate'] = list()
    #         output['ncell'] = list()
    #         for key in coordinate_dict.keys():
    #             output[key] = list()
    #
    #         idx = KSimage.label2idx(label_map)
    #         del idx[0]
    #
    #         for i, cell_idx in enumerate(idx.keys()):
    #             idx_pair = np.unravel_index(idx[cell_idx], label_map.shape)
    #             output['label'].append(cell_idx)
    #             output['coordinate'].append(np.mean(idx_pair,axis = 1))
    #
    #             ncell = stats.mode(ncell_map[idx_pair])[0]
    #             output['ncell'].append(ncell)
    #
    #             for key in coordinate_dict.keys():
    #                 signal = canvas[key][idx_pair]
    #                 output[key].append(np.sum(signal))
    #
    #         if output['label']:
    #             output['label'] = np.vstack(output['label'])
    #             output['coordinate'] = np.vstack(output['coordinate'])
    #             output['ncell'] = np.vstack(output['ncell'])
    #             output['green'] = np.vstack(output['green'])
    #             output['red'] = np.vstack(output['red'])
    #         else:
    #             output['label'] = []
    #             output['coordinate'] = []
    #             output['ncell'] = []
    #             output['green'] = []
    #             output['red'] = []
    #
    #         matlab.save(savename,{'output':output})
    #
    #     duration = time.time() - start_time
    #     print('Finish counting signals from sample %d of %d samples (%.2f)' % (iobj+1, nobj, duration))

    data = list()
    for iobj in range(nobj):
        data.append((iobj, dict_obj, result_path, nobj))

    mem = psutil.virtual_memory()
    npools = np.int(np.floor((mem.available / float(1024 ** 3)) / 2.8796))

    if npools > multiprocessing.cpu_count() - 2:
        npools = multiprocessing.cpu_count() - 2

    try:
        p = multiprocessing.Pool(npools)
        p.map(para_count_signal, data)
    finally:
        p.close()
        p.join()


#######################################################################################
def generate_spreadsheet(signal_count_result_path, spreadsheet_result_path):
    routine.create_dir(spreadsheet_result_path)

    file_list = glob.glob(os.path.join(signal_count_result_path, '*.mat'))
    for i, file in enumerate(file_list):

        start_time = time.time()
        basename = os.path.split(file)[1]
        basename = os.path.splitext(basename)[0]
        basename = basename + '.csv'
        savename = os.path.join(spreadsheet_result_path, basename)

        if not os.path.isfile(savename):

            output = matlab.load(file)['output']
            label = output[0]['label']
            coordinate = output[0]['coordinate']
            ncell = output[0]['ncell']
            size = output[0]['size']
            green = output[0]['green']
            red = output[0]['red']
            yellow = output[0]['yellow']

            csv_row = [['label', 'x', 'y', 'n cells', 'total size', 'total green', 'total red', 'total yellow']]
            for lab, coor, n, s, g, r, y in zip(label, coordinate, ncell, size, green, red, yellow):
                csv_row.append([lab[0], int(coor[0]), int(coor[1]), n[0], s[0], g[0], r[0], y[0]])

            KScsv.write_csv(csv_row, savename)

        duration = time.time() - start_time
        print('Finish writing result spread sheet of sample %d out of %d samples (%.2f sec)' %
              (i + 1, len(file), duration))


#######################################################################################
def retouch_coordinate(fish, coordinate):
    if coordinate.shape[0] > 6:
        nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(coordinate)
        _, indices = nbrs.kneighbors(coordinate)

        val = np.zeros(shape=indices.shape)
        for i in range(indices.shape[0]):
            for j in range(indices.shape[1]):
                try:
                    val[i, j] = np.max(fish[coordinate[indices[i, j], 0] - 3:coordinate[indices[i, j], 0] + 2,
                                       coordinate[indices[i, j], 1] - 3:coordinate[indices[i, j], 1] + 2])
                except:
                    val[i, j] = np.max(fish[coordinate[indices[i, j], 0], coordinate[indices[i, j], 1]])

        # truncated lower 10% under standard normal assumption
        lower = np.mean(val[:, 1:], axis=1) - 1.28 * np.std(val[:, 1:], axis=1)
        # truncated lower 15% under standard normal assumption
        # lower = np.mean(val[:, 1:], axis=1) - 1.04 * np.std(val[:, 1:], axis=1)
        # truncated lower 20% under standard normal assumption
        # lower = np.mean(val[:, 1:], axis=1) - 0.84 * np.std(val[:, 1:], axis=1)
        flag = val[:, 0] >= lower
        coordinate_new = coordinate[indices[flag, 0], :]
    else:
        coordinate_new = coordinate

    return coordinate_new


#######################################################################################
def spatial_adjust_coordinate_green(registered_fish_path, coordinate_path_green, coordinate_path_green_new):
    dict_path = {'fish': registered_fish_path,
                 'green': coordinate_path_green}

    dict_ext = {'fish': '.png',
                'green': '.mat'}

    obj_dict = get_pair_list(dict_path, dict_ext)

    nobjs = len(obj_dict[obj_dict.keys()[0]])

    routine.create_dir(coordinate_path_green_new)

    for i in range(nobjs):
        start_time = time.time()
        basename = os.path.split(obj_dict['green'][i])[1]
        basename = os.path.splitext(basename)[0]
        basename = basename + '.mat'
        savename = os.path.join(coordinate_path_green_new, basename)

        if not os.path.isfile(savename):
            fish = KSimage.imread(obj_dict['fish'][i])
            coordinate = matlab.load(obj_dict['green'][i])['coordinate']

            if coordinate.size:
                retouched_coordinate = retouch_coordinate(fish[:, :, 1], coordinate)
            else:
                retouched_coordinate = coordinate

            matlab.save(savename, {'coordinate': retouched_coordinate})

        duration = time.time() - start_time
        print('Finish removing false green signal in %d sample out of %d images (%.2f)' % (i + 1, nobjs, duration))


#######################################################################################
def spatial_adjust_coordinate_red(registered_fish_path, coordinate_path_red, coordinate_path_red_new):
    dict_path = {'fish': registered_fish_path,
                 'red': coordinate_path_red}

    dict_ext = {'fish': '.png',
                'red': '.mat'}

    obj_dict = get_pair_list(dict_path, dict_ext)

    nobjs = len(obj_dict[obj_dict.keys()[0]])

    routine.create_dir(coordinate_path_red_new)

    for i in range(nobjs):
        start_time = time.time()
        basename = os.path.split(obj_dict['red'][i])[1]
        basename = os.path.splitext(basename)[0]
        basename = basename + '.mat'
        savename = os.path.join(coordinate_path_red_new, basename)

        if not os.path.isfile(savename):
            fish = KSimage.imread(obj_dict['fish'][i])
            coordinate = matlab.load(obj_dict['red'][i])['coordinate']

            if coordinate.size:
                retouched_coordinate = retouch_coordinate(fish[:, :, 0], coordinate)
            else:
                retouched_coordinate = coordinate

            matlab.save(savename, {'coordinate': retouched_coordinate})

        duration = time.time() - start_time
        print('Finish removing false red signal in %d sample out of %d images (%.2f)' % (i + 1, nobjs, duration))


#######################################################################################
def spatial_adjust_coordinate_yellow(registered_fish_path, coordinate_path_yellow, coordinate_path_yellow_new):
    dict_path = {'fish': registered_fish_path,
                 'yellow': coordinate_path_yellow}

    dict_ext = {'fish': '.png',
                'yellow': '.mat'}

    obj_dict = get_pair_list(dict_path, dict_ext)

    nobjs = len(obj_dict[obj_dict.keys()[0]])

    routine.create_dir(coordinate_path_yellow_new)

    for i in range(nobjs):
        start_time = time.time()
        basename = os.path.split(obj_dict['yellow'][i])[1]
        basename = os.path.splitext(basename)[0]
        basename = basename + '.mat'
        savename = os.path.join(coordinate_path_yellow_new, basename)

        if not os.path.isfile(savename):
            fish = KSimage.imread(obj_dict['fish'][i])
            coordinate = matlab.load(obj_dict['yellow'][i])['coordinate']

            if coordinate.size:
                retouched_coordinate = retouch_coordinate(fish[:, :, 0], coordinate)
            else:
                retouched_coordinate = coordinate

            matlab.save(savename, {'coordinate': retouched_coordinate})

        duration = time.time() - start_time
        print('Finish removing false yellow signal in %d sample out of %d images (%.2f)' % (i + 1, nobjs, duration))
