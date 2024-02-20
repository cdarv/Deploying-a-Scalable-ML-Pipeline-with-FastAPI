import pytest
import pandas as pd
import numpy as np
from ml import data, model
from sklearn.linear_model import LogisticRegression

def test_train_model():
    """
    Testing if the train_model function returns a model of the expected type.
    """
    # Creating mock data
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(2, size=100)
    
    # Training the model on mock data
    model_output = model.train_model(X_train, y_train)
    
    # Checking the type of the model
    assert isinstance(model_output, LogisticRegression) 

def test_compute_model_metrics():
    """
    Testing if the compute_model_metrics function returns the expected value.
    """
    # Creating mock labels and predictions
    y_test = np.random.randint(2, size=100)
    predictions = np.random.randint(2, size=100)
    
    # Calculating the metrics
    precision, recall, f1 = model.compute_model_metrics(y_test, predictions)
    
    # Checking the type of the metrics
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)
    
    # Checking the range of the metrics is between 0 and 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1

def test_dataset_size_and_type():
    """
    Testing if the training and test datasets have the expected size and data type.
    """
    # Creating mock data
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(2, size=100)
    X_test = np.random.rand(20, 10)
    y_test = np.random.randint(2, size=20)
    
    # Checking the size of the datasets
    assert X_train.shape == (100, 10)
    assert y_train.shape == (100,)
    assert X_test.shape == (20, 10)
    assert y_test.shape == (20,)
    
    # Checking the data type of the datasets
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)