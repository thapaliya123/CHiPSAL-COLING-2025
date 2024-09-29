from enum import Enum
# Define an Enum for CSV path options
class CSVPathOption(Enum):
    TRAIN = 'train'
    VALID = 'valid'

class SheetNames(Enum):
    NEPALI = 'nepali'
    HINDI = 'hindi'
