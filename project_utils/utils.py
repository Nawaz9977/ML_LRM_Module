# utils/utils.py
import joblib

#file_path='boston_housing.csv'

def load_data(file_path):
    return pd.read_csv(file_path)

def save_model(model, model_path):
    joblib.dump(model, model_path)
