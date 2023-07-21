---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Initial Exploration

This notebook contains the initial exploration of the data sets.


## Setup & Read the Images

```python
%load_ext autoreload
%autoreload 2
```

```python
# Allow imports from `..`
import os
import sys

module_path = os.path.abspath("..")
if module_path not in sys.path:
    sys.path.append(module_path)
```

```python
import cv2
import numpy as np

from src.io import *
from src.match import *
from src.ops import *
from src.viz import *
```

```python
biden = FlyDatasetReader("../data/2023-03-10_biden", max_size = 1_500)
blank = biden.read_blank()
```

```python
blank.shape
```

```python
plot_image(blank, grayscale = True)
```

```python
img0 = biden.read_fly(0)
plot_image(img0)
```

```python
shiny = FlyDatasetReader("../data/2023-03-27_shiny", max_size = 1_500)
s_blank = shiny.read_blank()
```

```python
plot_image(s_blank)
```

## Template Matching

Template matching searches for one image (the "template") within another. For
this problem, the template is a manually cropped ruler from one of the fly
apparatus photos.

```python
template = blank[910:980, 270:1290]
plot_image(template)
```

```python
blank2 = cv2.imread(str(biden.blank_paths[0]))
blank2.shape
```

```python
template2 = blank2[3600:4000, 1000:5200]
print(template2.shape)
plot_image(template2)
```

```python
loc = extract_match(blank2, template2, cv2.TM_CCOEFF)

ax = plot_image(blank2)
plot_box(loc, ax)
```

```python
loc = extract_match(blank, template, cv2.TM_SQDIFF)

ax = plot_image(blank)
plot_box(loc, ax)
```

```python
loc = extract_match(img0, template, cv2.TM_SQDIFF)

ax = plot_image(img0)
plot_box(loc, ax)
```

How well does this work in an image with different lighting conditions?

```python
img23 = biden.read_fly(23)
loc = extract_match(img23, template, cv2.TM_SQDIFF)

ax = plot_image(img23)
plot_box(loc, ax)
```

What about images from a different data set?

```python
loc = extract_match(s_blank, template, cv2.TM_SQDIFF)

ax = plot_image(s_blank)
plot_box(loc, ax)
```

Unsurprisingly, template matching doesn't generalize well to images with very
different orientations. We can use image registration to try to orient the
image correctly, or just insist that the user specify the orientation (or
always provide images in a certain orientation).

What if the image has correct orientation?

```python
s_blank_rotated = cv2.rotate(s_blank, cv2.ROTATE_180)

loc = extract_match(s_blank_rotated, template, cv2.TM_SQDIFF)

ax = plot_image(s_blank_rotated)
plot_box(loc, ax)
```

With the correct orientation, template matching works well for these two
apparatuses, regardless of lighting. It might not generalize well to other
apparatuses, but if the number of different apparatuses is fairly small then
that's not necessarily a problem.

That said, other methods like image registration or edge detection may be more
robust.

After locating the ruler, the code will also need to locate positions along
the ruler. A simple approach is to locate the end marks and interpolate based
on the the 27 cm end-to-end length.


## Image Registration

```python
grayed = cv2.cvtColor(s_blank, cv2.COLOR_RGB2GRAY)
```

## Color Masking

Can we detect the rulers based on their color? They're white so let's mask
anything in the image that's not white.

```python
# Convert to hue-saturation-value to make this easier.
s_blank_hsv = cv2.cvtColor(s_blank, cv2.COLOR_RGB2HSV)
```

```python
blurred = cv2.GaussianBlur(s_blank_hsv, (0, 0), 1)
white = cv2.inRange(blurred, (0, 0, 180), (255, 24, 220))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
white = cv2.dilate(white, kernel)
plot_image(white, figsize = (10, 10))
```

```python
edged = cv2.Canny(white, 50, 150)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
edged = cv2.dilate(edged, kernel)
plot_image(edged, figsize = (10, 10))
```

```python
contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

```python
sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
contoured_all = []
for i in range(9):
    contoured = cv2.drawContours(s_blank.copy(), sorted_contours, i, (0, 255, 0), 1)
    contoured_all.append(contoured)
```

```python
chunked = [contoured_all[i:i + 3] for i in range(0, len(contoured_all), 3)]
plot_image_grid(*chunked, figsize = (12, 8))
```

```python
plot_image_grid([contoured_all[0]], figsize = (12, 8))
```

```python
lines = cv2.HoughLines(edged, 1, np.pi / 180, 250, None, 0, 0)
```

```python
# Plotting code from:
#    https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
#
cdst = s_blank.copy()
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(cdst, pt1, pt2, (0, 255, 0), 3, cv2.LINE_AA)

plot_image(cdst, figsize = (10, 10))
```

## Edge Detection

Edge detection methods locate sudden changes in the pixels of an image, which
often correspond to edges of physical objects.

```python
#plot_image(highpass_filter(grayed, (1001, 1001)))
laplaced = cv2.convertScaleAbs(cv2.Laplacian(grayed, cv2.CV_16S, (3, 3)))
plot_image(laplaced, figsize = (10, 10))
```

```python
# Blur to remove noise (but this can also make edges less distinct).
blurred = cv2.GaussianBlur(grayed, (201, 201), 0)

# NOTE: Gaussian blur is a kind of low-pass filter, it removes high-frequency
# details. A high-pass fiter might make it easier to detect the edges of the
# fly arenas and the ruler tick marks.
unsharped = unsharp_mask(grayed, 2.0)

plot_image_grid([grayed, unsharped], figsize = (15, 10))
```

```python
blurred2 = cv2.GaussianBlur(unsharped, (5, 5), 0)
# NOTE: The two thresholds for Canny edge detection are the potential-edge
# threshold and the sure-edge threshold. Potential edges are only edges if they
# are connected to sure edges.
edged = cv2.Canny(blurred2, 20, 150)
#plot_image(edged, figsize = (15, 15))

# NOTE: One way to clean up the contours is to use erosion/dilation operations
# on the detected edges before contour detection.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#eroded = cv2.erode(edged, kernel)
morphed = cv2.dilate(edged, kernel)
#dilated = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel)
#closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
plot_image(morphed, figsize = (10, 10))
```

```python
contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# NOTE: The drawContours function modifies its first argument.
contoured = cv2.drawContours(s_blank.copy(), contours, -1, (0, 255, 0), 1)
plot_image(contoured, figsize = (10, 10))
```

```python
plot_image(contoured, figsize = (20, 20))
```

```python
sorted_contours = sorted(contours, key = cv2.contourArea, reverse = True)
```

```python
# NOTE: It's probably safe to assume the rulers will cross the vertical midline
# of the image, so contours that don't do that can be ignored.
contoured = cv2.drawContours(s_blank.copy(), sorted_contours, 0, (0, 255, 0), 1)
plot_image(contoured, figsize = (14, 14))
```

```python
_, foo = cv2.threshold(grayed, 160, 255, cv2.THRESH_BINARY)
plot_image(foo, figsize = (10, 10))
```

## Other Exploration

```python
blank.shape
```

```python
print(f"Image size in memory: {blank.nbytes / 1024**2:.3} MB")
```

```python
plot_image(img0 - blank)
```

The flies are visible in the difference of image 0 with the blank, but there's
also a lot of background noise.
