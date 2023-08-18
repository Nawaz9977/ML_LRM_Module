# pipeline/Data_Injestion.py
import pandas as pd

class DataInjestion:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        return pd.read_csv(self.data_path)
