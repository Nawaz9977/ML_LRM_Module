# This file created expllicitily just to load already created model in pipeline folder and making prediction
# by providing differnet values(Features) as new_data--Sami Created this for his reference
import joblib

def load_model(model_path):
    loaded_model = joblib.load(model_path)
    return loaded_model

def make_predictions(model, new_data):
    predictions = model.predict(new_data)
    return predictions

if __name__ == "__main__":
    # Load the trained model
    model_path = '../pipeline/linear_regression_model.pkl'  # Adjust the path as needed
    loaded_model = load_model(model_path)

    # Example input data (features)
    new_data = [[2.0, 23.0, 6.0, 0, 0.5, 6.5, 50.0, 8.0, 5, 200, 7.0, 230, 09.0]]

    # Make predictions using the loaded model
    predictions = make_predictions(loaded_model, new_data)

    print("Predicted MEDV:", predictions)
