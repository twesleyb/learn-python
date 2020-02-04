#!/usr/bin/env python

# Use the argparse module to pass command line arguments to a python function.
import argparse

# Create the argparse object.
ap = argparse.ArgumentParser(description='''Put a description of your function here.''')

# Add a required argument.
ap.add_argument("duration", type = int, help="video duration")

# Optional arguments with defaults
ap.add_argument("-o", "--output", default=output,
            help="path to output video file; default = os.cwd")

ap.add_argument("-pi", "--picamera", type=int, default=1,
            help="whether or not the Raspberry Pi camera should be used; default = True")

ap.add_argument("-f", "--fps", type=int, default=30,
            help="FPS of output video; default = 45 fps")

ap.add_argument("-c", "--codec", type=str, default="h264",
            help="codec of output video; default = h264")

ap.add_argument("-p", "--preview", type=int, default=0,
                        help="preview output")

ap.add_argument("-a", "--annotate", type=int, default=1,
                        help="annotate video with timestamp")

# Collect the arguments.
args = vars(ap.parse_args())

[arg for arg in args]
