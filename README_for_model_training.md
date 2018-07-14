## Data preparation

Create the following folder structure

```
/Data
	/image
	/groundtruth
	/weight
	/group
```
* `/Data/image`: image
* `/Data/groundtruth`: place your segmentation groundtruth here. The current implementation require a user to prepare the groundtruth image such that
	* background pixel = 0
	* cell pixel = 1
	* boundary pixel = 2
* `/Data/weight`: place pre-computed weight images here. A weight image $$\mathbf{W}$$ is used to weight the loss value, i.e. $$\mathrm{loss}_j = \mathrm{loss}_j + w_j$$ for each pixel $$j$$ (Look at line 148 of `tf_model.py` for the implementation). If no weight is required, you can create a weight image where all pixels have zero value.
* `/Data/group`: For each image, create a csv file containing an integer index representing a group of the image. The idea here is to make sure that images from different groups will present in training, validation, and test split.
* Note that corresponding image, groundtruth, weight, and group files **must have the same file name**.

## Setting parameters

Parameters used for training the model can be set via variable `flags` in `train_model.py`

## Data augmentation
Look at `process_image_and_label` function in `tf_model_input.py`

## Loss
Look at `loss` function in `tf_model.py`

## Rescaling an image during test time
Look at `read_data_test` function in `tf_model_input_test.py`