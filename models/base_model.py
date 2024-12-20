# Teslapredictor/models/base_model.py
class BaseModel:
    def predict(self, data):
        raise NotImplementedError("This method should be overridden by subclasses")
