"""
Utils module initialization
"""
from .suppress_warnings import SuppressOutput
from .predictor import (
    predict_image, 
    find_image_in_uploads, 
    display_prediction_results,
    get_random_image_from_uploads,
    get_random_image_from_dataset
)

__all__ = [
    'SuppressOutput',
    'predict_image',
    'find_image_in_uploads',
    'display_prediction_results',
    'get_random_image_from_uploads',
    'get_random_image_from_dataset'
]