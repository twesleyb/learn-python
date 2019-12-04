#!/usr/bin/env python3
""" Day 1
String interpolation
"""

def interpolate(word1, word2):
    """ Print a sentance with different types of string interpolation."""

    # 1.
    print("First word: {} - Second word: {}".format(word1,word2))

    # 2. Equivalent to:
    print("First word: {0} - Second word: {1}".format(word1,word2))

    # 3. Equivalent to:
    print("First word: {first} - Second word: {second}".format(first=word1,second=word2))

    # 4. Equivalent to:
    word_list = [word1,word2]
    print("First word: {0[0]} - Second word: {0[1]}".format(word_list))

    # 5. Equivalent to: Pass input words as dictionary.
    print("First word: %(word1)s - Second word: %(word2)s" % {'word1':word1,
        'word2':word2})

    # 6. Equivalent to: Pass input words as tuple.
    print("First word: %s - Second word: %s" % (word1,word2))


if __name__ == '__main__':
    print("Interpolating 6 different ways:")
    interpolate("Ruby","GO")
