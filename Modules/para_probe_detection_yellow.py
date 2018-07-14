import sys
import os

sys.path.append(os.getcwd())

from KS_lib.general import KScsv

from KS_lib.tf_model_probe_detection import tf_model_main as main_probe_detection
from Modules.flags_probe_detection_yellow import flags as flags_probe_detection_yellow


def main(argv):
    file_list = argv[0]
    result_path = argv[1]
    he_dcis_segmentation_result_path = argv[2]
    igpu = argv[3]

    row_list = KScsv.read_csv(file_list)
    main_probe_detection.main(1, 'test_model', flags_probe_detection_yellow,
                              row_list, result_path,
                              he_dcis_segmentation_result_path, igpu)


if __name__ == "__main__":
    main(sys.argv[1:])
