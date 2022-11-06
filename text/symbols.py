'''
Defines the set of symbols used in text input to the model.

The default is a Korean characters from U+11xx.
You can check the code using 'ord' function in Python3. '''

_pad     = '_'
_sos     = '^'
_eos     = '~'
_special = '-'
_punctuation = '!\'(),.:;? '

_jamo_leads  = "".join(chr(c) for c in range(0x1100, 0x1113))
_jamo_vowels = "".join(chr(c) for c in range(0x1161, 0x1176))
_jamo_tails  = "".join(chr(c) for c in range(0x11a8, 0x11c3))
_letters = _jamo_leads + _jamo_vowels + _jamo_tails

symbols = [_pad] + [_sos] + [_eos] + list(_special) + list(_punctuation) + list(_letters)

