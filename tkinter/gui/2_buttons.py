#!/usr/bin/env python3
''' Creating a button widget.'''

# Tkinter in two steps:
# 0. Initialize tkinter.
# 1. Create a widget.
# 2. Display on screen.

import tkinter as tk

root = tk.Tk()
myButton = tk.Button(root, text="Click me!",padx=50,pady=50)
myButton.pack()
root.mainloop()


