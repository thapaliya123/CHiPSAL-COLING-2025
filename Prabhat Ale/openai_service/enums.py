from enum import Enum
# Define an Enum for CSV path options
class CSVPathOption(Enum):
    TRAIN = 'train'
    VALID = 'valid'

class SheetNames(Enum):
    NEPALI_TRAIN = 'train_nepali'
    HINDI_TRAIN = 'train_hindi'
    NEPALI_VALID = 'valid_nepali'
    HINDI_VALID = 'valid_hindi'
