import sys
import os
sys.path.append(os.getcwd())

from KS_lib.general import KScsv

from KS_lib.tf_model_he_dcis_segmentation import tf_model_main as main_he_dcis_segmentation
from Modules.flags_he_dcis_segmentation import flags as flags_he_dcis_segmentation

def main(argv):
    he_log_file = argv[0]
    he_dcis_segmentation_result_path = argv[1]
    igpu = argv[2]

    row_list = KScsv.read_csv(he_log_file)
    main_he_dcis_segmentation.main(1, 'test_model', flags_he_dcis_segmentation,
                                                  row_list, he_dcis_segmentation_result_path, igpu)

if __name__ == "__main__":
   main(sys.argv[1:])