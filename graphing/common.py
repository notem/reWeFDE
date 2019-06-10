
# Timing feature list
#FEATURE_CATEGORIES = [
#    ('Time', (1, 24)),
#    ('Pkt. per\n Second', (25, 150)),
#    ('IBD-LF', (151, 170)),
#    ('IBD-OFF', (171, 190)),
#    ('IBD-FF', (191, 210)),
#    ('IMD', (211, 230)),
#    ('Variance', (231, 250)),
#    ('MED', (251, 270)),
#    ('Burst\nLength', (271, 290)),
#    ('IBD-IFF', (291, 310)),
#]

# WeFDE feature list
FEATURE_CATEGORIES = [
    ('Pkt. Count', (1, 13)),
    ('Time', (14, 37)),
    ('Ngram', (38, 161)),
    ('Transposition', (162, 765)),
    ('Interval-I', (766, 1365)),
    ('Interval-II', (1366, 1967)),
    ('Interval-III', (1968, 2553)),
    ('Pkt. Distribution', (2554, 2778)),
    ('Burst', (2779, 2789)),
    ('First 20', (2790, 2809)),
    ('First 30', (2810, 2811)),
    ('Last 30', (2812, 2813)),
    ('Pkt. per Second', (2814, 2939)),
    ('CUMUL', (2940, 3043))
]
"""
Define names and ranges for feature categories.
"""


COLORS = [
    'blue',
    'green',
    'cyan',
    'red',
    'purple',
    'yellow',
    'olive',
    'orange',
    'violet'
    'teal',
]
"""
Ordered list of colors to use when graphing.
"""
