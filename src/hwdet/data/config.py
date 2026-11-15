"""Tunable numbers of this package, one section per module."""

# detds
REUSE = 3
FORM_REUSE = 2
BLANK_FORMS = 2
PAPER = {
    "blank": 0.28,
    "lined": 0.20,
    "squared": 0.12,
    "margin": 0.08,
    "spiral": 0.08,
    "seyes": 0.08,
    "dotted": 0.08,
    "legal": 0.08,
}
MARGIN = 0.05
SKEW = ((0.7, 0.0, 0.4), (0.2, 1.0, 3.0), (0.1, 3.0, 8.0))
VAL_EVERY = 10

# ds3k
DS3K_SKEW = ((0.75, 0.0, 0.3), (0.2, 0.3, 1.0), (0.05, 1.0, 1.8))
FILL = (0.50, 1.00)
MIN_ASPECT = 1.15
MAX_SOURCE_SKEW = 4.0
SHARES = (("form", 9), ("hand", 9), ("notes", 9), ("print", 6))
