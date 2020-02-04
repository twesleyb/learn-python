#!/usr/bin/env Python

# Use a namedtuple to quickly create a class.
from collections import namedtuple
car = namedtuple('car', 'color mileage')

# Our new "car" class:
my_car = car('red', 3812.4)

my_car.color

my_car.mileage

my_car

# namedtuples, like tuples, are immutable!
my_car.color = "blue"


