"""
extract_patches.py includes different methods for patches extraction to prepare experimental data
"""
########################################################################################################################
import numpy as np
import collections

########################################################################################################################
def sliding_window(dict_obj, size_input_patch, size_output_patch, stride):
    #key_list = ['HE', 'DAPI', 'weight']
    key_list = list(dict_obj.keys()) # all items except last item: 'group'

    # padding
    for key in dict_obj.keys():
        padrow = size_input_patch[0]
        padcol = size_input_patch[1]

        if dict_obj[key].ndim == 2:
            dict_obj[key] = np.lib.pad(dict_obj[key], ((padrow, padrow), (padcol, padcol)), 'symmetric')
        else:
            dict_obj[key] = np.lib.pad(dict_obj[key], ((padrow, padrow), (padcol, padcol), (0, 0)), 'symmetric')

    ntimes_row = int(np.floor((dict_obj[key_list[0]].shape[0] - size_input_patch[0]) / float(stride[0])) + 1)
    ntimes_col = int(np.floor((dict_obj[key_list[0]].shape[1] - size_input_patch[1]) / float(stride[1])) + 1)
    row_range = range(0, ntimes_row * stride[0], stride[0])
    col_range = range(0, ntimes_col * stride[1], stride[1])

    assert size_input_patch[0] >= size_output_patch[0]
    assert size_input_patch[1] >= size_output_patch[1]

    centre_index_row = int(round((size_input_patch[0] - size_output_patch[0]) / 2.0))
    centre_index_col = int(round((size_input_patch[1] - size_output_patch[1]) / 2.0))

    for row in row_range:
        for col in col_range:
            coord_dict = dict()
            out_obj_dict = dict()

            for key in dict_obj.keys():
                if dict_obj[key].ndim == 2:
                    if key == key_list[0]:
                        out_obj_dict[key] = dict_obj[key][
                                            row : row + size_input_patch[0],
                                            col : col + size_input_patch[1]]
                    else:
                        out_obj_dict[key] = dict_obj[key][
                                            row + centre_index_row: row + centre_index_row + size_output_patch[0],
                                            col + centre_index_col: col + centre_index_col + size_output_patch[1]]
                    coord_dict[key] = row, col
                else:
                    if key == key_list[0]:
                        out_obj_dict[key] = dict_obj[key][
                                            row : row + size_input_patch[0],
                                            col : col + size_input_patch[1], :]
                    else:
                        out_obj_dict[key] = dict_obj[key][
                                            row + centre_index_row: row + centre_index_row + size_output_patch[0],
                                            col + centre_index_col: col + centre_index_col + size_output_patch[1], :]
                    coord_dict[key] = row, col

            yield (out_obj_dict, coord_dict)

########################################################################################################################
def coordinate(dict_obj, dict_patch_size, coordinate):
    list_keys = dict_obj.keys()

    coordinate = np.around(coordinate)
    coordinate = coordinate.astype(np.int32)

    temp = []
    for key in list_keys:
        temp.append(dict_patch_size[key][0:2])

    for index, (pointx, pointy) in enumerate(coordinate):

        dict_patches = dict()
        coord_dict = dict()

        for key in list_keys:
            obj = dict_obj[key]
            size_input_patch = dict_patch_size[key]

            centre_index_row = int(round((size_input_patch[0] - 1) / 2.0))
            centre_index_col = int(round((size_input_patch[1] - 1) / 2.0))

            if obj.ndim == 2:
                temp = obj[int(pointx - centre_index_row): int(pointx + centre_index_row + 1),
                                           int(pointy - centre_index_col): int(pointy + centre_index_col + 1)]
                dict_patches[key] = temp[0:size_input_patch[0],0:size_input_patch[1]]
                coord_dict[key] = pointx - centre_index_row, pointy - centre_index_col
            else:
                temp = obj[int(pointx - centre_index_row): int(pointx + centre_index_row + 1),
                                          int(pointy - centre_index_col): int(pointy + centre_index_col + 1), :]
                dict_patches[key] = temp[0:size_input_patch[0],0:size_input_patch[1],:]
                coord_dict[key] = pointx - centre_index_row, pointy - centre_index_col

        yield (dict_patches, coord_dict)

########################################################################################################################
def coordinate_jittering(dict_obj, dict_patch_size, coordinate):

    list_keys = dict_obj.keys()

    coordinate = np.around(coordinate)
    coordinate = coordinate.astype(np.int32)

    temp = []
    for key in list_keys:
        temp.append(dict_patch_size[key][0:2])
    size_input_patch = np.min(temp,axis = 0)

    centre_index_row_smallest = int(round((size_input_patch[0] - 1) / 2.0))
    centre_index_col_smallest = int(round((size_input_patch[1] - 1) / 2.0))

    for index, (pointx, pointy) in enumerate(coordinate):

        dict_patches = dict()
        coord_dict = dict()

        r_row = np.random.randint(-centre_index_row_smallest, centre_index_row_smallest+1, 1)[0]
        r_col = np.random.randint(-centre_index_col_smallest, centre_index_col_smallest+1, 1)[0]

        pointx += r_row
        pointy += r_col

        for key in list_keys:
            obj = dict_obj[key]
            size_input_patch = dict_patch_size[key]

            centre_index_row = int(round((size_input_patch[0] - 1) / 2.0))
            centre_index_col = int(round((size_input_patch[1] - 1) / 2.0))

            if obj.ndim == 2:
                temp = obj[int(pointx - centre_index_row): int(pointx + centre_index_row + 1),
                                           int(pointy - centre_index_col): int(pointy + centre_index_col + 1)]
                dict_patches[key] = temp[0:size_input_patch[0],0:size_input_patch[1]]
                coord_dict[key] = pointx - centre_index_row, pointy - centre_index_col
            else:
                temp = obj[int(pointx - centre_index_row): int(pointx + centre_index_row + 1),
                                          int(pointy - centre_index_col): int(pointy + centre_index_col + 1), :]
                dict_patches[key] = temp[0:size_input_patch[0],0:size_input_patch[1],:]
                coord_dict[key] = pointx - centre_index_row, pointy - centre_index_col

        yield (dict_patches, coord_dict)

########################################################################################################################
def coordinate_jittering_with_radius(dict_obj, dict_patch_size, coordinate, radius):

    list_keys = dict_obj.keys()

    coordinate = np.around(coordinate)
    coordinate = coordinate.astype(np.int32)

    temp = []
    for key in list_keys:
        temp.append(dict_patch_size[key][0:2])

    for index, (pointx, pointy) in enumerate(coordinate):

        dict_patches = dict()
        coord_dict = dict()

        r_row = np.random.randint(-radius, radius+1, 1)[0]
        r_col = np.random.randint(-radius, radius+1, 1)[0]

        pointx += r_row
        pointy += r_col

        for key in list_keys:
            obj = dict_obj[key]
            size_input_patch = dict_patch_size[key]

            centre_index_row = int(round((size_input_patch[0] - 1) / 2.0))
            centre_index_col = int(round((size_input_patch[1] - 1) / 2.0))

            if obj.ndim == 2:
                temp = obj[int(pointx - centre_index_row): int(pointx + centre_index_row + 1),
                                           int(pointy - centre_index_col): int(pointy + centre_index_col + 1)]
                dict_patches[key] = temp[0:size_input_patch[0],0:size_input_patch[1]]
                coord_dict[key] = pointx - centre_index_row, pointy - centre_index_col
            else:
                temp = obj[int(pointx - centre_index_row): int(pointx + centre_index_row + 1),
                                          int(pointy - centre_index_col): int(pointy + centre_index_col + 1), :]
                dict_patches[key] = temp[0:size_input_patch[0],0:size_input_patch[1],:]
                coord_dict[key] = pointx - centre_index_row, pointy - centre_index_col

        yield (dict_patches, coord_dict)
