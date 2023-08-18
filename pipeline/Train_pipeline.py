# pipeline/Train_pipeline.py

import os
import sys

# Append the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Data_Injestion import DataInjestion
from Data_Transformation import DataTransformation
from model_trainer import ModelTrainer
from Predict_pipeline import PredictionPipeline
from project_utils.utils import save_model

class TrainPipeline:
    def __init__(self, data_path):
        self.data_injestion = DataInjestion(data_path)
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run(self):
        data = self.data_injestion.load_data()
        X, y = self.data_transformation.preprocess_data(data)
        self.model_trainer.train_model(X, y)
        trained_model = self.model_trainer.model
        save_model(trained_model, 'linear_regression_model.pkl')

if __name__ == "__main__":
    train_pipeline = TrainPipeline(data_path='boston_housing.csv')
    train_pipeline.run()
