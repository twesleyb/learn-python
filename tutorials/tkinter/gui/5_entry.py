#!/usr/bin/env python3
''' Recieving user input.'''

import tkinter as tk

# Intialized widget.
root = tk.Tk()

# Widget name.
name = tk.Label(root,text="Please enter your name:")
name.pack()

# Collect user input.
myInput = tk.Entry(root)
myInput.pack()

# Function to respond to a click.
def myClick():
    response = "Hello " + myInput.get() + "!"
    myLabel = tk.Label(root, text = response + "\n Remember to not fuck it up!")
    myLabel.pack()

# Create button.
myButton = tk.Button(root, text="Submit", command = myClick)
myButton.pack()

# Run tkinter.
root.mainloop()
