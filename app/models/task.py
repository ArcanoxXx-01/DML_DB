from enum import Enum

class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"