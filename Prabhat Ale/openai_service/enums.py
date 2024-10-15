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
    VALID_NEPALI_TO_HINDI = 'valid_nepali_to_hindi'
    VALID_HINDI_TO_NEPALI = 'valid_hindi_to_nepali'
    CONFUSING_NEPALI_TWEET = 'confusing_nepali_tweet'
    CONFUSING_HINDI_TWEET = 'confusing_hindi_tweet'
