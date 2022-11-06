""" modified for Korean language
Folked from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that ru over the input text at both training and eval time.

Cleaners can be selected by passian a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Most of cleaners are Korean-specific.
'''

import re
from jamo import h2j

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def basic_cleaners(text):
    '''Basic pipeline that support Korean-only'''
    text = collapse_whitespace(text)
    text = h2j(text)
    return text

