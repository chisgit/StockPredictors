# Teslapredictor/models/linear_model.py
from models.base_model import BaseModel

class LinearModel(BaseModel):
    def predict(self, data):
        return f"Linear model prediction: {data}"
