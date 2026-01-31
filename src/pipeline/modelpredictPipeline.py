import pickle 
import numpy as np
from src.utils.read_yml import readYaml
from pathlib import Path
import pandas as pd
class Predict:
    def __init__(self):
        pass

    def predict(self, test_data):

        config = readYaml("src/config/config.yml")

        model_path = Path(config["model"]["model_pickle_path"])
        with open(model_path,"rb") as f:
            model = pickle.load(f)

        precessor_path = Path(config["data"]["transformation"]["pickle_obj_path"])
        with open(precessor_path,"rb") as f:
            precessor = pickle.load(f)

        columns = [
            "Age",
            "Gender",
            "Department",
            "CGPA",
            "Sleep_Duration",
            "Study_Hours",
            "Social_Media_Hours",
            "Physical_Activity",
            "Stress_Level"
        ]

        df = pd.DataFrame([test_data], columns=columns)

        scaled_data = precessor.transform(df)

        y_pred = model.predict(scaled_data)

        return y_pred