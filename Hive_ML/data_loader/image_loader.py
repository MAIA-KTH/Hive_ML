from os import PathLike
from pathlib import Path
from typing import Union, List, Tuple

import SimpleITK as sitk
from SimpleITK import Image


def get_3D_image_sequence_list_from_4D_image(image_filename: Union[str, PathLike]) -> List[Image]:
    """
    Convert a single 4D Image into a list of 3D Images.
    Each item of the list is one sequence of the 4D image in ITK image format with the same spacing and origin of the main image.

    Parameters
    ----------
    image_filename :
        4D image file path.

    Returns
    -------
        List of 3D Images (one per sequence).

    """
    itk_image = sitk.ReadImage(image_filename)
    img_array = sitk.GetArrayFromImage(itk_image)

    n_sequence = itk_image.GetSize()[-1]
    direction_3D = itk_image.GetDirection()[:3] + itk_image.GetDirection()[4:7] + itk_image.GetDirection()[8:11]
    origin = itk_image.GetOrigin()[:3]
    spacing = itk_image.GetSpacing()[:3]

    sitk_3D_image_sequences = []
    for sequence in range(n_sequence):
        img_sequence = img_array[sequence]
        itk_img_sequence = sitk.GetImageFromArray(img_sequence)
        itk_img_sequence.SetOrigin(origin)
        itk_img_sequence.SetSpacing(spacing)
        itk_img_sequence.SetDirection(direction_3D)
        sitk_3D_image_sequences.append(itk_img_sequence)

    return sitk_3D_image_sequences


def get_id_label(filename, config_dict) -> Tuple[str, str]:
    """
    Function to provide Subject ID and Label ID for a given file path,  extracting the information from the parent folder tree.

    Parameters
    ----------
    filename :
        File path to extract Subject ID and Label.
    config_dict :
        Configuration dictionary containing the mapping for each label class to the corresponding label ID.

    Returns
    -------
            Subject ID and Label ID.

    """
    subject_ID = Path(filename).parent.name
    label_name = Path(filename).parent.parent.name
    label_dict = config_dict["label_dict"]
    label = None
    for label_key in label_dict:
        if label_dict[label_key] == label_name:
            label = label_key

    return subject_ID, label
