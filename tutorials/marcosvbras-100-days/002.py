#!/usr/bin/env python3
""" Day 2
Exploring lists with append, insert and del.
"""

def add(characters):
    """ Add elements to a list. """
    characters.append("Tyler")
    characters.append("Christina")
    characters.append("Tom")
    print("new list: {}".format(characters))
    
def insert(characters,index):
    """ Insert a character into list."""
    characters.insert(index,"Foobar")


if __name__ == '__main__':
    print("Interpolating 6 different ways:")
    interpolate("Ruby","GO")
