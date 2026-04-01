from enum import Enum


class ActionType(str, Enum):
    # Easy
    DROP_DUPLICATES = "DROP_DUPLICATES"
    DROP_COLUMN = "DROP_COLUMN"
    FILL_NA = "FILL_NA"

    # Medium
    STRIP_CHAR = "STRIP_CHAR"
    TYPE_CAST = "TYPE_CAST"
    LOWERCASE = "LOWERCASE"

    # Hard
    VALUE_COUNTS = "VALUE_COUNTS"
    MAP_VALUES = "MAP_VALUES"
