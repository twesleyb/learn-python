#!/usr/bin/env python3

# tkinter - Python's built-in GUI
# everything in tkinter is a widget. 

import tkinter as tk

# create a root widge before doing anything else!
root = tk.Tk()

# Create a label widget.
mylabel = tk.Label(root, text="Hello World!")

# Shove it onto the screen with pack.
mylabel.pack()

# Loop to show widget until program ends.
root.mainloop()
