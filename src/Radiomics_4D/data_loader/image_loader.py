from os import PathLike
from pathlib import Path
from typing import Union, Dict, Any, List

import SimpleITK as sitk
from SimpleITK import Image


def get_3D_image_sequence_list_from_4D_image(image_filename: Union[str, PathLike]) -> List[Image]:
    """
    Parameters
    ----------
    image_filename  : Union[str, PathLike]
        Image filename for the 4D volume.

    Returns
    -------
    List[Image]
        Each item of the list is one sequence of the 4D image in
        ITK image format with the same spacing, orientation and origin of the main
        image.

    """
    itk_image = sitk.ReadImage(image_filename)
    n_sequence = itk_image.GetSize()[-1]
    img_array = sitk.GetArrayFromImage(itk_image)
    direction_3D = itk_image.GetDirection()[:3] + itk_image.GetDirection()[4:7] + itk_image.GetDirection()[8:11]
    sitk_3D_image_sequences = []
    for sequence in range(n_sequence):
        img_sequence = img_array[sequence]
        itk_img_sequence = sitk.GetImageFromArray(img_sequence)
        itk_img_sequence.SetOrigin(itk_image.GetOrigin()[:3])
        itk_img_sequence.SetSpacing(itk_image.GetSpacing()[:3])
        itk_img_sequence.SetDirection(direction_3D)
        sitk_3D_image_sequences.append(itk_img_sequence)

    return sitk_3D_image_sequences


def get_id_label(filename: Union[str, PathLike], config_dict: Dict[str, Any]) -> (str, str):
    """
    Get Subject ID and corresponding label for the given filename.

    Parameters
    ----------
    filename : Union[str, PathLike]
        Image filename.
    config_dict : Dict[str, Any]
        Dictionary including Experiment and Dataset configuration parameters.

    Returns
    -------
    (str, str)
        Subject ID (the parent folder name) and numerical label (extracted from the Label dict in the Config dict).
    """
    subject_ID = str(Path(filename).parent.name)
    label_name = Path(filename).parent.parent.name
    label_dict = config_dict["label_dict"]
    label = None
    for label_key in label_dict:
        if label_dict[label_key] == label_name:
            label = label_key

    return subject_ID, label
