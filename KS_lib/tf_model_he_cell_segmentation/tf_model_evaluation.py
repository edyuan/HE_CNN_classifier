import os
from KS_lib.general import KScsv
from KS_lib.general import matlab
from KS_lib.image import KSimage
import numpy as np
from sklearn import metrics
import time

###############################################################################################
def evaluate(object_folder):
    test_images_list = os.path.join(object_folder, 'test_images_list.csv')
    test_labels_list = os.path.join(object_folder, 'test_labels_list.csv')

    test_image_filenames = KScsv.read_csv(test_images_list)
    test_label_filenames = KScsv.read_csv(test_labels_list)

    all_prediction = list()
    all_label = list()
    f1score_per_image = list()
    all_score = list()
    for i_image,(image_file, label_file) in enumerate(zip(test_image_filenames, test_label_filenames)):

        tick = time.time()

        basename = os.path.basename(image_file[0])
        basename = os.path.splitext(basename)[0]
        image_file = os.path.join(object_folder, 'result', basename + '.mat')

        # Read in result and label
        mat_content = matlab.load(image_file)
        score = mat_content['mask']
        prediction = score > 0.5
        prediction = prediction.astype('float')

        label = KSimage.imread(label_file[0])
        label = label.astype('float')
        label = label/255.0
        label = label > 0.5
        label = label.astype('float')

        prediction = np.reshape(prediction,-1)
        label = np.reshape(label,-1)
        score = np.reshape(score, -1)

        all_prediction.append(prediction)
        all_label.append(label)
        all_score.append(score)

        f1score = metrics.f1_score(label, prediction, average='binary')
        f1score_per_image.append(f1score)

        duration = time.time() - tick
        print('evaluate %d / %d (%.2f sec)' %( i_image + 1, len(test_image_filenames), duration))

    all_label = np.reshape(np.array(all_label),-1)
    all_prediction = np.reshape(np.array(all_prediction),-1)
    all_score = np.reshape(np.array(all_score), - 1)

    total_f1score = metrics.f1_score(all_label, all_prediction, average = 'binary')
    avg_f1score = np.mean(f1score_per_image)
    average_precision = metrics.average_precision_score(all_label, all_score, average = 'micro')

    return total_f1score, avg_f1score, average_precision, f1score_per_image






