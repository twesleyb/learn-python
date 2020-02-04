#!/usr/bin/env python3

#------------------------------------------------------------------------------
# ## Lambda function tutorial.
#------------------------------------------------------------------------------
# From: https://www.w3schools.com/python/python_lambda.asp


# First, define a simple function that asks the user if they wish to proceed.

import re
import sys

def query(question):
    answer = input(question + " [Y]es/[N]o: ")
    
    if re.match('^[Yy]', answer):
        pass
    else:
        sys.exit()


# Define lamda functions.
print('''
        Lambda functions are small anonymous functions.
        A lambda function can take any number of arguments, 
        but can only have one expresion.
        ''')

query("Continue?")

# Example 1. 
print('''
    Syntax: lambda arguments : expression
    For Example:
    x = lambda a : a + 10
    x(a = 5)
    ''')

x = lambda a : a + 10
print(f"x(5) = {x(5)}")

print('''
    This is equivalent to the function:
    def x(a):
        return(a+10)
    ''')

#------------------------------------------------------------------------------
# ## More examples.
#------------------------------------------------------------------------------

query("See another example?")

print('''
        Lambda functions can take multiple arguements!
        x = lambda a,b : a * b
        print(x(5,6))
        ''')

x = lambda a,b : a * b
c = x(5,6)
print(f"x(5,6) = {c}")

print('''
        For more examples, including how to use lambda functions
        within other functions, see:
        https://www.w3schools.com/python/python_lambda.asp
        ''')

