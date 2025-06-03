"""
Customer Behavior Analysis Package

This package provides tools for analyzing customer behavior and predicting
purchase patterns using big data and machine learning techniques.
"""

__version__ = "1.0.0"
__author__ = "Shree BD"

from .data_processing import DataProcessor
from .model_training import ModelTrainer

__all__ = ["DataProcessor", "ModelTrainer"] 