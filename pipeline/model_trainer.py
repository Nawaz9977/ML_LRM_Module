# pipeline/model_trainer.py
from sklearn.linear_model import LinearRegression

class ModelTrainer:
    def __init__(self):
        self.model = LinearRegression()

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
