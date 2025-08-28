import os
from dataclasses import dataclass
from enum import IntEnum
from typing import List, NamedTuple

import cv2 as cv
import numpy as np
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import UID, generate_uid

from rt_utils.utils import ROIData


class SliceDirections(NamedTuple):
    row_direction: np.ndarray
    column_direction: np.ndarray
    slice_direction: np.ndarray


@dataclass
class FrameInfo:
    sop_class_uid: UID
    sop_instance_uid: UID

    instance_number: int
    slice_location: float
    image_position_patient: list[float]

    @classmethod
    def from_single_frame(cls, ds: Dataset) -> "FrameInfo":
        slice_location = getattr(ds, "SliceLocation", get_slice_position(ds))

        return cls(
            sop_class_uid=ds.SOPClassUID,
            sop_instance_uid=ds.SOPInstanceUID,
            instance_number=ds.InstanceNumber,
            slice_location=slice_location,
            image_position_patient=ds.ImagePositionPatient,
        )

    @classmethod
    def from_multi_frame(cls, ds: Dataset, index: int) -> "FrameInfo":
        per_frame = ds.PerFrameFunctionalGroupsSequence[index]

        return cls(
            sop_class_uid=per_frame[(0x0020, 0x9172)][0].ReferencedSOPClassUID,
            sop_instance_uid=per_frame[(0x0020, 0x9172)][0].ReferencedSOPInstanceUID,
            instance_number=per_frame[(0x0020, 0x9171)][0].InstanceNumber,
            slice_location=per_frame[(0x0020, 0x9171)][0].SliceLocation,
            image_position_patient=per_frame[(0x0020, 0x9113)][0].ImagePositionPatient,
        )


@dataclass
class StudyInfo:
    study_date: str
    series_date: str
    study_time: str
    series_time: str
    study_description: str
    series_description: str
    study_instance_uid: UID
    series_instance_uid: UID
    study_id: str
    series_number: int

    @classmethod
    def from_dataset(cls, ds: Dataset) -> "StudyInfo":
        return cls(
            study_date=ds.StudyDate,
            series_date=getattr(ds, "SeriesDate", ""),
            study_time=ds.StudyTime,
            series_time=getattr(ds, "SeriesTime", ""),
            study_description=getattr(ds, "StudyDescription", ""),
            series_description=getattr(ds, "SeriesDescription", ""),
            study_instance_uid=ds.StudyInstanceUID,
            series_instance_uid=ds.SeriesInstanceUID,
            study_id=ds.StudyID,
            series_number=ds.SeriesNumber,
        )


@dataclass
class PatientInfo:
    patient_name: str
    patient_id: str
    patient_birth_date: str
    patient_sex: str
    patient_age: str
    patient_size: str
    patient_weight: str

    @classmethod
    def from_dataset(cls, ds: Dataset) -> "PatientInfo":
        return cls(
            patient_name=getattr(ds, "PatientName", ""),
            patient_id=getattr(ds, "PatientID", ""),
            patient_birth_date=getattr(ds, "PatientBirthDate", ""),
            patient_sex=getattr(ds, "PatientSex", ""),
            patient_age=getattr(ds, "PatientAge", ""),
            patient_size=getattr(ds, "PatientSize", ""),
            patient_weight=getattr(ds, "PatientWeight", ""),
        )


@dataclass
class DicomInfo:
    image_orientation_patient: list[float]
    image_position_patient: list[float]
    pixel_spacing: list[float]
    slice_spacing: float
    slice_directions: SliceDirections

    rows: int
    columns: int

    number_of_frames: int
    frame_info: list[FrameInfo]
    frame_of_reference_uid: UID
    # rt_referenced_study_sequence: Sequence

    study_info: StudyInfo
    patient_info: PatientInfo

    @property
    def sop_instance_uids(self) -> set[UID]:
        return {f.sop_instance_uid for f in self.frame_info}

    @classmethod
    def from_single_frame(cls, ds_list: list[Dataset]) -> "DicomInfo":
        first = ds_list[0]
        return cls(
            image_orientation_patient=first.ImageOrientationPatient,
            image_position_patient=first.ImagePositionPatient,
            pixel_spacing=first.PixelSpacing,
            slice_spacing=get_spacing_between_slices(ds_list),
            slice_directions=get_slice_directions(first),
            rows=first.Rows,
            columns=first.Columns,
            number_of_frames=len(ds_list),
            frame_info=[FrameInfo.from_single_frame(ds) for ds in ds_list],
            frame_of_reference_uid=getattr(first, 'FrameOfReferenceUID', generate_uid()),
            study_info=StudyInfo.from_dataset(first),
            patient_info=PatientInfo.from_dataset(first),
        )

    @classmethod
    def from_multi_frame(cls, ds: Dataset) -> "DicomInfo":
        frame_info = [
            FrameInfo.from_multi_frame(ds, i) for i in range(ds.NumberOfFrames)
        ]

        return cls(
            image_orientation_patient=ds.ImageOrientationPatient,
            image_position_patient=frame_info[0].image_position_patient,
            pixel_spacing=ds.PixelSpacing,
            slice_spacing=get_spacing_between_slices(ds),
            slice_directions=get_slice_directions(ds),
            rows=ds.Rows,
            columns=ds.Columns,
            number_of_frames=ds.NumberOfFrames,
            frame_info=frame_info,
            frame_of_reference_uid=getattr(ds, 'FrameOfReferenceUID', generate_uid()),
            study_info=StudyInfo.from_dataset(ds),
            patient_info=PatientInfo.from_dataset(ds),
        )


def load_sorted_image_series(dicom_series_path: str) -> DicomInfo:
    """
    File contains helper methods for loading / formatting DICOM images and contours
    """
    if not os.path.exists(dicom_series_path):
        raise FileNotFoundError(f"Input path {dicom_series_path} does not exist")

    if os.path.isfile(dicom_series_path):
        ds = dcmread(dicom_series_path)
        if hasattr(ds, "NumberOfFrames"):
            return DicomInfo.from_multi_frame(ds)
        else:
            raise RuntimeError("Input is a single file, but not a multi-frame DICOM")
    else:
        ds = load_dcm_images_from_path(dicom_series_path)

        if hasattr(ds[0], "SliceLocation"):
            # Sort by slice location if available
            ds.sort(key=lambda x: x.SliceLocation)
        else:
            # Fallback to sorting by instance number
            ds.sort(key=get_slice_position)

        return DicomInfo.from_single_frame(ds)


def load_dcm_images_from_path(dicom_series_path: str) -> List[Dataset]:
    series_data: List[Dataset] = []
    for root, _, files in os.walk(dicom_series_path):
        for file in files:
            try:
                ds = dcmread(os.path.join(root, file))
                if hasattr(ds, "pixel_array"):
                    series_data.append(ds)

            except Exception:
                # Not a valid DICOM file
                continue

    return series_data


def get_contours_coords(roi_data: ROIData, series_data: DicomInfo):
    transformation_matrix = get_pixel_to_patient_transformation_matrix(series_data)

    series_contours = []
    for i in range(series_data.number_of_frames):
        mask_slice = roi_data.mask[:, :, i]

        # Do not add ROI's for blank slices
        if np.sum(mask_slice) == 0:
            series_contours.append([])
            continue

        # Create pin hole mask if specified
        if roi_data.use_pin_hole:
            mask_slice = create_pin_hole_mask(mask_slice, roi_data.approximate_contours)

        # Get contours from mask
        contours, _ = find_mask_contours(mask_slice, roi_data.approximate_contours)
        validate_contours(contours)

        # Format for DICOM
        formatted_contours = []
        for contour in contours:
            # Add z index
            contour = np.concatenate(
                (np.array(contour), np.full((len(contour), 1), i)), axis=1
            )

            transformed_contour = apply_transformation_to_3d_points(
                contour, transformation_matrix
            )
            dicom_formatted_contour = np.ravel(transformed_contour).tolist()
            formatted_contours.append(dicom_formatted_contour)

        series_contours.append(formatted_contours)

    return series_contours


def find_mask_contours(mask: np.ndarray, approximate_contours: bool):
    approximation_method = (
        cv.CHAIN_APPROX_SIMPLE if approximate_contours else cv.CHAIN_APPROX_NONE
    )
    contours, hierarchy = cv.findContours(
        mask.astype(np.uint8), cv.RETR_TREE, approximation_method
    )
    # Format extra array out of data
    contours = list(
        contours
    )  # Open-CV updated contours to be a tuple so we convert it back into a list here
    for i, contour in enumerate(contours):
        contours[i] = [[pos[0][0], pos[0][1]] for pos in contour]
    hierarchy = hierarchy[0]  # Format extra array out of data

    return contours, hierarchy


def create_pin_hole_mask(mask: np.ndarray, approximate_contours: bool):
    """
    Creates masks with pin holes added to contour regions with holes.
    This is done so that a given region can be represented by a single contour.
    """

    contours, hierarchy = find_mask_contours(mask, approximate_contours)
    pin_hole_mask = mask.copy()

    # Iterate through the hierarchy, for child nodes, draw a line upwards from the first point
    for i, array in enumerate(hierarchy):
        parent_contour_index = array[Hierarchy.parent_node]
        if parent_contour_index == -1:
            continue  # Contour is not a child

        child_contour = contours[i]

        line_start = tuple(child_contour[0])

        pin_hole_mask = draw_line_upwards_from_point(
            pin_hole_mask, line_start, fill_value=0
        )
    return pin_hole_mask


def draw_line_upwards_from_point(
    mask: np.ndarray, start, fill_value: int
) -> np.ndarray:
    line_width = 2
    end = (start[0], start[1] - 1)
    mask = mask.astype(np.uint8)  # Type that OpenCV expects
    # Draw one point at a time until we hit a point that already has the desired value
    while mask[end] != fill_value:
        cv.line(mask, start, end, fill_value, line_width)

        # Update start and end to the next positions
        start = end
        end = (start[0], start[1] - line_width)
    return mask.astype(bool)


def validate_contours(contours: list):
    if len(contours) == 0:
        raise Exception(
            "Unable to find contour in non empty mask, please check your mask formatting"
        )


def get_pixel_to_patient_transformation_matrix(series_data: DicomInfo):
    """
    https://nipy.org/nibabel/dicom/dicom_orientation.html
    """

    offset = np.array(series_data.image_position_patient)
    row_spacing, column_spacing = series_data.pixel_spacing
    slice_spacing = series_data.slice_spacing
    row_direction, column_direction, slice_direction = series_data.slice_directions

    mat = np.identity(4, dtype=np.float32)
    # The following might appear counter-intuitive, i.e. multiplying the row direction with the column spacing and vice-versa
    # But is the correct way to create the transformation matrix, see https://nipy.org/nibabel/dicom/dicom_orientation.html
    mat[:3, 0] = row_direction * column_spacing
    mat[:3, 1] = column_direction * row_spacing
    mat[:3, 2] = slice_direction * slice_spacing
    mat[:3, 3] = offset

    return mat


def get_patient_to_pixel_transformation_matrix(series_data: DicomInfo):
    offset = np.array(series_data.image_position_patient)
    row_spacing, column_spacing = series_data.pixel_spacing
    slice_spacing = series_data.slice_spacing
    row_direction, column_direction, slice_direction = series_data.slice_directions

    # M = [ rotation&scaling   translation ]
    #     [        0                1      ]
    #
    # inv(M) = [ inv(rotation&scaling)   -inv(rotation&scaling) * translation ]
    #          [          0                                1                  ]

    # The following might appear counter-intuitive, i.e. dividing the row direction with the column spacing and vice-versa
    # But is the correct way to create the inverse transformation matrix, see https://nipy.org/nibabel/dicom/dicom_orientation.html
    linear = np.identity(3, dtype=np.float32)
    linear[0, :3] = row_direction / column_spacing
    linear[1, :3] = column_direction / row_spacing
    linear[2, :3] = slice_direction / slice_spacing

    mat = np.identity(4, dtype=np.float32)
    mat[:3, :3] = linear
    mat[:3, 3] = offset.dot(-linear.T)

    return mat


def apply_transformation_to_3d_points(
    points: np.ndarray, transformation_matrix: np.ndarray
):
    """
    * Augment each point with a '1' as the fourth coordinate to allow translation
    * Multiply by a 4x4 transformation matrix
    * Throw away added '1's
    """
    vec = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    return vec.dot(transformation_matrix.T)[:, :3]


def get_slice_position(series_slice: Dataset):
    if hasattr(series_slice, "SliceLocation"):
        return series_slice.SliceLocation

    _, _, slice_direction = get_slice_directions(series_slice)
    return np.dot(slice_direction, series_slice.ImagePositionPatient)


def get_slice_directions(series_slice: Dataset) -> SliceDirections:
    orientation = series_slice.ImageOrientationPatient
    row_direction = np.array(orientation[:3])
    column_direction = np.array(orientation[3:])
    slice_direction = -np.cross(row_direction, column_direction)

    if not np.allclose(
        np.dot(row_direction, column_direction), 0.0, atol=1e-3
    ) or not np.allclose(np.linalg.norm(slice_direction), 1.0, atol=1e-3):
        raise Exception("Invalid Image Orientation (Patient) attribute")

    return SliceDirections(row_direction, column_direction, slice_direction)


def get_spacing_between_slices(series_data: Dataset | list[Dataset]) -> float:
    if isinstance(series_data, list):
        if hasattr(series_data[0], "SpacingBetweenSlices"):
            return float(series_data[0].SpacingBetweenSlices)

        if len(series_data) < 2:
            # Return nonzero value for one slice just to make the transformation matrix invertible
            return 1.0

        first = get_slice_position(series_data[0])
        last = get_slice_position(series_data[-1])
        return (last - first) / (len(series_data) - 1)

    if series_data.NumberOfFrames > 1:
        if hasattr(series_data, "SpacingBetweenSlices"):
            return float(series_data.SpacingBetweenSlices)

        return (
            series_data.PerFrameFunctionalGroupsSequence[1][(0x0020, 0x9171)][
                0
            ].SliceLocation
            - series_data.PerFrameFunctionalGroupsSequence[0][(0x0020, 0x9171)][
                0
            ].SliceLocation
        )

    # Return nonzero value for one slice just to make the transformation matrix invertible
    return 1.0


def create_series_mask_from_contour_sequence(
    series_data: DicomInfo, contour_sequence: Sequence
):
    mask = create_empty_series_mask(series_data)
    transformation_matrix = get_patient_to_pixel_transformation_matrix(series_data)

    # Iterate through each slice of the series, If it is a part of the contour, add the contour mask
    for i, frame in enumerate(series_data.frame_info):
        slice_contour_data = get_slice_contour_data(frame, contour_sequence)
        if len(slice_contour_data):
            mask[:, :, i] = get_slice_mask_from_slice_contour_data(
                series_data, slice_contour_data, transformation_matrix
            )
    return mask


def get_slice_contour_data(series_slice: FrameInfo, contour_sequence: Sequence):
    slice_contour_data = []

    # Traverse through sequence data and get all contour data pertaining to the given slice
    for contour in contour_sequence:
        for contour_image in contour.ContourImageSequence:
            if contour_image.ReferencedSOPInstanceUID == series_slice.sop_instance_uid:
                slice_contour_data.append(contour.ContourData)

    return slice_contour_data


def get_slice_mask_from_slice_contour_data(
    series_data: DicomInfo, slice_contour_data, transformation_matrix: np.ndarray
):
    # Go through all contours in a slice, create polygons in correct space and with a correct format
    # and append to polygons array (appropriate for fillPoly)
    polygons = []
    for contour_coords in slice_contour_data:
        reshaped_contour_data = np.reshape(
            contour_coords, [len(contour_coords) // 3, 3]
        )
        translated_contour_data = apply_transformation_to_3d_points(
            reshaped_contour_data, transformation_matrix
        )
        polygon = [np.around([translated_contour_data[:, :2]]).astype(np.int32)]
        polygon = np.array(polygon).squeeze()
        polygons.append(polygon)
    slice_mask = create_empty_slice_mask(series_data).astype(np.uint8)
    cv.fillPoly(img=slice_mask, pts=polygons, color=1)
    return slice_mask


def create_empty_series_mask(series_data: DicomInfo):
    mask_dims = (
        int(series_data.columns),
        int(series_data.rows),
        series_data.number_of_frames,
    )
    mask = np.zeros(mask_dims).astype(bool)
    return mask


def create_empty_slice_mask(series_data: DicomInfo):
    mask_dims = (int(series_data.columns), int(series_data.rows))
    mask = np.zeros(mask_dims).astype(bool)
    return mask


class Hierarchy(IntEnum):
    """
    Enum class for what the positions in the OpenCV hierarchy array mean
    """

    next_node = 0
    previous_node = 1
    first_child = 2
    parent_node = 3
