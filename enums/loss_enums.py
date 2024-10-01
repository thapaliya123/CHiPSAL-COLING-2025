from enum import Enum

class LossFuncEnum(Enum):
    BINARY_CROSSENTROPY = "binary_crossentropy"
    CATEGORICAL_CROSSENTROPY = "categorical_crossentropy"
    WEIGHTED_CROSSENTROPY = "weighted_crossentropy"
    FOCAL_CROSSENTROPY = "focal_crossentropy"
