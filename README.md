# CNN for Classification of Breast Cancer Histology Images
* built using Korsuk's HE Cell Seg Framework
* classifies Patch Images into an arbitrary number of classes
  * add classes using flags['n_classes']
  * class labels are encoded by adding prefix 'classnumber_' to file names. for example, for a class 1 image, append '1_' to '01_420_2_5_24.png' to get '1_01_420_2_5_24.png'
* add folders by creating a Data/ folder in HE_CNN_classifier/Data:
  * add each image types as Data/FolderName
  * for example, 'Data/HE', 'Data/Keratin', 'Data/DAPI'
  * add corresponding flag in flags['dict_path'] and flags['dict_ext']
* group information is currently unused
* model currently only runs inference model on H&E images in the 'HE' folder
