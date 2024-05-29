import numpy as np

class DemoModel:
    def __init__(self) -> None:
        # whatever need to be initialize
        pass
    def predict(self, query: str) -> 'np.ndarray':
        # return 1d ndarray
        result = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        return result