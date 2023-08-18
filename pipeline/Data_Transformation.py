# pipeline/Data_Transformation.py
from sklearn.preprocessing import StandardScaler

class DataTransformation:
    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess_data(self, data):
        X = data.drop(columns=['MEDV'])  # Drop target column
        y = data['MEDV']  # Target column

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y
