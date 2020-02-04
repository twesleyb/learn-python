#!/usr/bin/env python3

# tkinter - Python's built-in GUI
# everything in tkinter is a widget. 

import tkinter as tk

# create a root widge before doing anything else!
root = tk.Tk()

# Create a label widget.
mylabel1 = tk.Label(root, text="Hello World!")
mylabel2 = tk.Label(root, text="My name is Tyler Bradshaw.")

# Use grid to display text.
mylabel1.grid(row=0,column=0) # Where will text be displayed?
mylabel2.grid(row=1,column=1)

# Loop to show widget until program ends.
root.mainloop()
