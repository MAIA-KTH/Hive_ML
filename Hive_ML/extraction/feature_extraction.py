from os import PathLike
from pathlib import Path
from typing import Union, List, Dict, Any

import SimpleITK as sitk
import numpy as np
import six

from Hive_ML.data_loader.image_loader import get_3D_image_sequence_list_from_4D_image, get_id_label


def extract_features_for_image_and_mask(extractor, image_filename: Union[str, PathLike, List[str]],
                                        mask_filename: Union[str, PathLike], config_dict: Dict[str, Any],
                                        distance_map_filename: Union[str, PathLike] = None, n_bins: int = 3) -> List[
    Dict[str, Any]]:
    """
    Extract Radiomics for a given image (or list of images), and the corresponding binary mask.
    If a distance map is specified, extract radiomics at each depth interval, specified by ``n_bins``.
    ``image_filename`` can be a 3D volume, a 4D volume or a list of 3D volumes.

    Example:
        with n_bins=3, Radiomics are extracted at 3 depth interval: 0-33%, 34-66% and 67-100% of the maximum depth.


    Parameters
    ----------
    extractor :
        Radiomics extractor object.
    image_filename :
        file path of the volume/s from where to extract Radiomics.
    mask_filename :
        Segmentation mask file path, used to extract Radiomics only on ROIs.
    config_dict :
        Configuration dictionary.
    distance_map_filename :
        Optional distance map file path, used to extract radiomics at different depth intervals.
    n_bins :
        number of intervals to extract depth-wise radiomics.

    Returns
    -------
    List of extracted features, one element in the list per each 3D volume.
    """
    if type(image_filename) is list:
        img = sitk.ReadImage(image_filename[0])
    else:
        img = sitk.ReadImage(image_filename)

    sitk_3D_image_sequence_list = []

    if type(image_filename) is list:
        for single_image_filename in image_filename:
            if Path(single_image_filename).is_file():
                img = sitk.ReadImage(single_image_filename)
            else:
                img = None
            sitk_3D_image_sequence_list.append(img)
    else:
        if img.GetMetaData('dim[0]') == '3':
            sitk_3D_image_sequence_list.append(img)
        else:
            sitk_3D_image_sequence_list = get_3D_image_sequence_list_from_4D_image(image_filename)

    sitk_mask = sitk.ReadImage(mask_filename)

    mask_array = sitk.GetArrayFromImage(sitk_mask)

    mask_array[mask_array > 0] = 1

    sitk_mask_thresholded = sitk.GetImageFromArray(mask_array)
    sitk_mask_thresholded.CopyInformation(sitk_mask)

    image_types_dict = extractor.enabledImagetypes
    image_types = [x.lower() for x in image_types_dict]

    features_sequence_list = []

    if type(image_filename) is list:
        subject_ID, label = get_id_label(image_filename[0], config_dict)
    else:
        subject_ID, label = get_id_label(image_filename, config_dict)

    if distance_map_filename is None:
        for sequence_number, itk_3D_image in enumerate(sitk_3D_image_sequence_list):
            if itk_3D_image is None:
                features_map = {"Subject_ID": subject_ID, "Subject_Label": label, "Sequence_Number": sequence_number}
            else:
                if itk_3D_image.GetSize() != sitk_mask.GetSize():
                    continue
                features = extractor.execute(itk_3D_image, sitk_mask_thresholded)
                features_map = {"Subject_ID": subject_ID, "Subject_Label": label, "Sequence_Number": sequence_number}
                for key, val in six.iteritems(features):
                    if key.startswith(tuple(image_types)):
                        features_map[key] = features[key]
            features_sequence_list.append(features_map)
    else:
        distance_map = sitk.GetArrayFromImage(sitk.ReadImage(distance_map_filename))
        max_depth = np.max(distance_map)
        for sequence_number, itk_3D_image in enumerate(sitk_3D_image_sequence_list):
            if itk_3D_image is None:
                features_map = {"Subject_ID": subject_ID, "Subject_Label": label, "Sequence_Number": sequence_number}
            else:
                features_map = {"Subject_ID": subject_ID, "Subject_Label": label, "Sequence_Number": sequence_number}
                for depth_range in np.arange(0, max_depth, max_depth / n_bins):
                    depth_interval = [depth_range, depth_range + max_depth / n_bins]
                    mask = sitk.GetArrayFromImage(sitk_mask_thresholded)
                    where = np.where(
                        (distance_map >= depth_interval[0]) & (distance_map <= depth_interval[1]) & (mask != 0), mask,
                        0)
                    if np.sum(where) == 0:
                        ...

                    else:
                        mask_image = sitk.GetImageFromArray(where)
                        mask_image.CopyInformation(itk_3D_image)
                        if itk_3D_image.GetSize() != sitk_mask.GetSize():
                            continue
                        features = extractor.execute(itk_3D_image, mask_image)
                        for key, val in six.iteritems(features):
                            if key.startswith(tuple(image_types)):
                                features_map[key + "_{}-{}".format(round(depth_interval[0] * 100 / max_depth, 1),
                                                                   round(depth_interval[1] * 100 / max_depth, 1))] = \
                                    features[key]
            features_sequence_list.append(features_map)
    return features_sequence_list


def extract_perfusion_feature(perfusion_feature_id: str, perfusion_map_filename: Union[str, PathLike, List[str]],
                              distance_map_filename: Union[str, PathLike, List[str]], subject: str,
                              config_dict: Dict[str, Any],
                              n_bins_list: List[int] = [2]) -> Dict[str, Any]:
    """
    Function to extract statistical features (mean, sd, median, max and min) for a given 3D perfusion map at different depth intervals.

    Parameters
    ----------
    perfusion_feature_id    :
        Perfusion Feature name.
    perfusion_map_filename  :
        Perfusion map filepath.
    distance_map_filename   :
        Distance map filepath.
    subject :
        Subject ID.
    config_dict :
        Configuration dictionary.
    n_bins_list :
        List with different depth intervals to consider in the feature extraction.

    Returns
    -------
        Dictionary of extracted statistic features at different depth intervals.
    """
    if "perfusion_depth_bins" in config_dict:
        n_bins_list = config_dict["perfusion_depth_bins"]

    feature_image = sitk.ReadImage(perfusion_map_filename)
    feature_map = sitk.GetArrayFromImage(feature_image)
    distance_map_image = sitk.ReadImage(distance_map_filename)
    distance_map = sitk.GetArrayFromImage(distance_map_image)
    feature_depth = {}
    feature_depth["Subject_ID"] = subject
    feature_depth["Subject_Label"] = get_id_label(perfusion_map_filename, config_dict)[1]
    feature_depth["Sequence_Number"] = 0
    max_depth = np.max(distance_map)
    for n_bins in n_bins_list:
        for depth_range in np.arange(0, max_depth, max_depth / n_bins):
            depth_interval = [depth_range, depth_range + max_depth / n_bins]
            where = np.where(
                (distance_map >= depth_interval[0]) & (distance_map <= depth_interval[1]) & (feature_map != 0), True,
                False)
            if feature_map[where].shape[0] == 0:
                section_mean = None
                section_min = None
                section_max = None
                section_std = None
                section_median = None
            else:
                section_mean = np.mean(feature_map[where])
                section_max = np.max(feature_map[where])
                section_min = np.min(feature_map[where])
                section_std = np.std(feature_map[where])
                section_median = np.median(feature_map[where])

            feature_depth[
                "avg_{}_{}-{}_depth".format(perfusion_feature_id, round(depth_interval[0] * 100 / max_depth, 1),
                                            round(depth_interval[1] * 100 / max_depth), 1)] = section_mean
            feature_depth[
                "median_{}_{}-{}_depth".format(perfusion_feature_id, round(depth_interval[0] * 100 / max_depth, 1),
                                               round(depth_interval[1] * 100 / max_depth), 1)] = section_median
            feature_depth[
                "sd_{}_{}-{}_depth".format(perfusion_feature_id, round(depth_interval[0] * 100 / max_depth, 1),
                                           round(depth_interval[1] * 100 / max_depth), 1)] = section_std
            feature_depth[
                "min_{}_{}-{}_depth".format(perfusion_feature_id, round(depth_interval[0] * 100 / max_depth, 1),
                                            round(depth_interval[1] * 100 / max_depth), 1)] = section_min
            feature_depth[
                "max_{}_{}-{}_depth".format(perfusion_feature_id, round(depth_interval[0] * 100 / max_depth, 1),
                                            round(depth_interval[1] * 100 / max_depth), 1)] = section_max

    return feature_depth
