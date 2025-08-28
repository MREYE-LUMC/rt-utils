import pytest

from rt_utils import ds_helper, image_helper


def test_correctly_acquire_optional_ds_field(series_path):
    series_data = image_helper.load_sorted_image_series(series_path)

    patient_age = series_data.patient_info.patient_age
    ds = ds_helper.create_rtstruct_dataset(series_data)

    assert ds.PatientAge == patient_age


def test_ds_creation_without_patient_age(series_path):
    # Ensure only 1 file in series data so ds_helper uses the file for header reference
    series = image_helper.load_dcm_images_from_path(series_path)

    # Remove optional field
    original_age = series[0]["PatientAge"].value
    del series[0]["PatientAge"]

    with pytest.raises(Exception):
        series[0]["PatientAge"]

    series_data = image_helper.DicomInfo.from_single_frame(series)
    ds = ds_helper.create_rtstruct_dataset(series_data)

    assert ds.PatientAge != original_age
    assert ds.PatientAge == ""
