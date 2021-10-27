from os import PathLike
from typing import Union, Dict, Any, List

import SimpleITK as sitk
import six
from radiomics.featureextractor import RadiomicsFeatureExtractor

from Radiomics_4D.data_loader.image_loader import get_3D_image_sequence_list_from_4D_image, get_id_label


def extract_features_for_image_and_mask(
    extractor: RadiomicsFeatureExtractor,
    image_filename: Union[str, PathLike],
    mask_filename: Union[str, PathLike],
    config_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    For a given image and mask pair, extracts features using the provided extractor.
    Parameters
    ----------
    extractor : RadiomicsFeatureExtractor
        Radiomics Feature extractor used to extract the features.
    image_filename : Union[str, PathLike]
        Image filename.
    mask_filename : Union[str, PathLike]
        Mask filename.
    config_dict : Dict[str, Any]
        Dictionary including Experiment and Dataset configuration parameters.

    Returns
    -------
    List[Dict[str,Any]]
        List of  Dictionaries, each one containing the extracted features in the format key (Feature Name) : value (Feature Value).
        Each dictionary of the list represents the extracted features for a single 3D sequence in the 4D volume.
    """  # noqa: E501
    sitk_3D_image_sequence_list = get_3D_image_sequence_list_from_4D_image(image_filename)
    sitk_mask = sitk.ReadImage(mask_filename)

    image_types_dict = extractor.enabledImagetypes
    image_types = [x.lower() for x in image_types_dict]

    features_sequence_list = []

    subject_ID, label = get_id_label(image_filename, config_dict)

    for sequence_number, itk_3D_image in enumerate(sitk_3D_image_sequence_list):
        features = extractor.execute(itk_3D_image, sitk_mask)
        features_map = {"Subject_ID": subject_ID, "Subject_Label": label, "Sequence_Number": sequence_number}
        for key, val in six.iteritems(features):
            if key.startswith(tuple(image_types)):
                features_map[key] = features[key]
        features_sequence_list.append(features_map)

    return features_sequence_list
