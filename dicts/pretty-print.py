#!/usr/bin/env python

# Use print.
my_dict = {'a' :  23, 'b' : 42, 'c' : 0xc0ffee}
print('This is a regular dictionary:')
print(my_dict)

# Use json.dumps() to pretty print python dictionaries.
print('The json module can make things look prettier:')
import json
print(json.dumps(my_dict, indent = 4, sort_keys = True))

print('See also, the built-in pprint function!')
