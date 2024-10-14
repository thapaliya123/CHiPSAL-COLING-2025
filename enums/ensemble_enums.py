from enum import Enum

class EnsembleEnum(Enum):
    SOFT_VOTE = "soft_vote"
    HARD_VOTE = "hard_vote"
    NO_ENSEMBLE = "no_ensemble"


def get_enum_values(enum_cls):
    return [e.value for e in enum_cls]