#!/usr/bin/env python3
# Usage example of the figure first python module.
# From: http://flyranch.github.io/figurefirst/
# Paper here: http://conference.scipy.org/proceedings/scipy2017/pdfs/lindsay.pdf

# Figure first requires:
# * Inkscape - an open source SVG graphics editor: https://inkscape.org/about/overview/
# * matplotlib

import numpy as np
import matplotlib.pyplot as plt
from figurefirst import FigureLayout

layout = FigureLayout('hello_world_layout.svg')
layout.make_mplfigures()
ax = layout.axes['ax_name']['axis']

x = np.arange(10)
y = x**2
ax.plot(x, y, lw=4)

layout.insert_figures('target_layer_name')
layout.write_svg('hello_world_output.svg')
