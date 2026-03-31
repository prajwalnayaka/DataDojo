from enum import Enum

class ActionType(str, Enum):
    DROP_DUPLICATES = "DROP_DUPLICATES"
    DROP_COLUMN = "DROP_COLUMN"
    FILL_NA = "FILL_NA"