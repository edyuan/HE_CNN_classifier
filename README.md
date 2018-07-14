# Nuclear segmentation from H&E

## Prerequisites
* Anaconda 4.4.0 (Python 2.7 version) https://www.tensorflow.org/versions/r0.12/get_started/os_setup
* tensorflow 1.4 
* h5py 2.7.0 http://docs.h5py.org/en/latest/build.html
* hdf5storage 0.1.14 https://github.com/frejanordsiek/hdf5storage
* numpy 1.12.1
* scipy 0.19.0
* opencv-python 3.2.0.7 https://anaconda.org/menpo/opencv3
* Pillow 4.1.1 http://pillow.readthedocs.io/en/3.4.x/installation.html
* psutil 5.2.2 https://github.com/giampaolo/psutil/blob/master/INSTALL.rst
* scikit-image 0.13.0 http://scikit-image.org/docs/dev/install.html
* scikit-learn 0.18.1 http://scikit-learn.org/stable/install.html
* maybe some other packages which I missed not including here but would show up as error when the program is runned anyway

## Installation (step-by-step)
1. Install **tensorflow 1.4** by creating an Anaconda virtual enviroment called **tensorflow**. Follow that instruction in this [documentation](https://www.tensorflow.org/install/install_linux).
2. Install other dependencies.
3. Download the code from this github repository.
4. Download the trained CNN models from this [link](https://www.dropbox.com/sh/bvzf82amvk4wxac/AABu37CFtGD3glvf1Kpu66o9a?dl=0) and place them in **./Modules** directory. You should have 
  * ./Modules/experiment_he_cell_segmentation
  * ./Modules/experiment_he_dcis_segmentation

## How to run the code
1. Create the following directories:
 * ./HE
2. Place H&E images inside **HE** folder.
3. Run main.py and the magic should happen
4. Results are in ./Result and ./Result_tumour folders 

