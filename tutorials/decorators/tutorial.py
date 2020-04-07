#!/usr/bin/env python3
''' Real Python decorator tutorial
https://realpython.com/primer-on-python-decorators/
'''

# A simple function:
def add_one(number):
    return number + 1

add_one(2)

# In Python functions are first class objects.
# Functions can be passed around and used as arguments.

# Standard functions:
def say_hello(name):
    return f"Hello {name}"

def be_awesome(name):
    return f"Yo {name}, together we are awesome!"

# The greet_bob function expects a function as an argument!
def greet_bob(greeter_func):
    return greeter_func("Bob")

# We can pass greet_bob a function, say_hello
greet_bob(say_hello) 

# This is the same as say_hello("Bob")

# Defining inner functions.
# A function in a function!

def parent():
    print("Printing from the parent() function.")
    def first_child():
        print("Printing from the first_child() function.")
    def second_child():
        print("Printing from the second_child() function.")
    first_child()
    second_child()
# Done.

parent()

# Returning functions from functions.
def parent(number):
    def first_child():
        return "Emma"
    def second_child():
        return "Liam"
    if number == 1:
        return first_child
    else:
        return second_child
# Done.

first = parent(1)
second = parent(2)
first()
second

# Simple decorators.
def my_decorator(func):
    def wrapper():
        print("Something is happening before my function is called.")
        func()
        print("Something after my function is called.")
    return wrapper
# EOF

def say_whee():
    print("Wheee!")

say_whee = my_decorator(say_whee)

say_whee()

# Decorators wrap a function and modify its behavior.

from datetime import datetime
def silence_at_night(func):
    def wrapper():
        if 7 <= datetime.now().hour < 22:
            func()
        else:
            pass # Supress output.
    return wrapper

say_whee = silence_at_night(say_whee)

say_whee() # Only works during the daytime!

# Simplier syntax: pie syntax "@"
@my_decorator
def poop():
    print("I took a big poo.")

poop()

# Check out the decorator function in decorators.py. 

from decorators import repeat

@repeat
def say_whee():
    print("Whee!")

say_whee()

# Try decorating a function with arguments.
@repeat
def greet(name):
    print(f"Hello {name}.")

greet("Why am I repeating myself")
