import sys
import os
sys.path.append(os.getcwd())

from KS_lib.general import KScsv

from KS_lib.tf_model_dcis_cell_segmentation import tf_model_main as main_dcis_cell_segmentation
from Modules.flags_dcis_cell_segmentation import flags as flags_dcis_cell_segmentation

def main(argv):
    file_list = argv[0]
    dcis_cell_segmentation_result_path = argv[1]
    he_dcis_segmentation_result_path = argv[2]
    igpu = argv[3]

    row_list = KScsv.read_csv(file_list)
    main_dcis_cell_segmentation.main(1, 'test_model', flags_dcis_cell_segmentation,
                                    row_list, dcis_cell_segmentation_result_path,
                                    he_dcis_segmentation_result_path, igpu)

if __name__ == "__main__":
    main(sys.argv[1:])