#!/usr/bin/env python3
''' Recieving user input.'''

import tkinter as tk

# Intialized widget.
root = tk.Tk()

# Collect user input.
myInput = tk.Entry(root,borderwidth=50)
myInput.pack()

# Function to respond to a click.
def myClick():
    myLabel = tk.Label(root, text ="Don't fuck it up!")
    myLabel.pack()

# Create button.
myButton = tk.Button(root, text="Click me!",
        padx=50,pady=50,command=myClick,fg="blue",bg="red")
myButton.pack()

# Run tkinter.
root.mainloop()
