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

# Registration Mark Experiments


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
from src.viz import *
```

```python
reg = FlyDatasetReader("../data/registration/", max_size = 1_500)
blank = reg.read_blank()
```

```python
blank.shape
```

```python
plot_image(blank, figsize = (10, 10))
```

## Preprocessing (Color Thresholding)

Hue-saturation-value (HSV) typically records hue as angles along a color wheel,
from 0 to 360 degrees. 0 is pure red, 120 is pure green, and 240 is pure blue.
In OpenCV, hue ranges from 0 to 179, which appears to just be a rescaling (to
fit within 8 bits).

```python
# Find green pixels.
def find_color_shapes(
    image, min_h = 60, max_h = 120, kernel_size = 5
):
    """Mask all pixels except those in a specific hue range.
    """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #mask = cv2.inRange(image_hsv, (min_h, 30, 127), (max_h, 255, 255))
    mask = cv2.inRange(image_hsv, (min_h, 127, 127), (max_h, 255, 255))

    # Use close op to close holes, then open op to remove noise.
    kernel_size = (kernel_size, kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


result = find_color_shapes(blank)
plot_image(result, figsize = (10, 10))
```

```python
def find_squares(mask, k = 3, min_area_pct = 0.0001, max_area_pct = 0.1):
    """Find k squares within a given mask.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape[:2]
    m_area = h * w
    print(f"{h=}, {w=}")

    # Drop contours with areas too small or large.
    # Compute contour areas and drop contours with area less than 1.
    area = np.array([cv2.contourArea(con) for con in contours])
    ok_area = (m_area * min_area_pct < area) & (area < m_area * max_area_pct)
    contours = [x for i, x in enumerate(contours) if ok_area[i]]
    area = area[ok_area]

    # Sort by area (largest to smallest).
    ix = np.argsort(-area)
    area = area[ix]
    contours = [contours[i] for i in ix]

    # TODO: check in case there are < 3 contours.

    # Find k contours with similar areas by finding the lowest-variance
    # k-sequence.
    std = [np.std(area[i:i+k]) for i in range(len(area) - k + 1)]
    ix = np.argmin(std)
    area = area[ix:ix+k]
    contours = contours[ix:ix+k]

    # Simplify the contours.
    # NOTE: We could also check that the simplified contours have 4 corners.
    # NOTE: Or check aspect ratio of bounding box is near 1.
    contours = [
        cv2.approxPolyDP(x, 0.05 * cv2.arcLength(x, True), True)
        for x in contours]

    contours = [np.squeeze(x) for x in contours]

    return contours, area
```

```python
# Use 3 detected squares to find the fly arena and transform it to a rectangle.
def arg_nearest_pair(x):
    ix = np.argsort(x)
    best = np.argmin(np.diff(x[ix]))
    return ix[best:best+2]


def find_edges(contours):
    # Determine which two boxes are aligned horizontally and which two are
    # aligned vertically.
    x_means = np.array([c[:, 0].mean() for c in contours])
    x_aligned = [contours[i] for i in arg_nearest_pair(x_means)]

    y_means = np.array([c[:, 1].mean() for c in contours])
    y_aligned = [contours[i] for i in arg_nearest_pair(y_means)]

    # Compute bottom-leftmost, top-leftmost, top-rightmost, bottom-rightmost
    # corners for these contours.
    # Get bottom-leftmost of each contour, then get bottom-leftmost of those.
    tl = contours[0][0, :]
    bl = contours[0][1, :]
    br = contours[0][2, :]
    tr = contours[0][3, :]
    for c in contours[1:]:
        if (c[0, :] < tl).any():
            tl = c[0, :]
        if c[1, 0] < bl[0] or c[1, 1] > bl[1]:
            bl = c[1, :]
        if c[2, 0] > br[0] or c[2, 1] > br[1]:
            br = c[2, :]
        if c[3, 0] > tr[0] or c[3, 1] < tr[1]:
            tr = c[3, :]
    return contours, np.stack([tl, bl, br, tr])

z, coords = find_edges(squares)
print(z)

contoured = cv2.drawContours(blank.copy(), [coords], -1, (0, 255, 0), 1)
plot_image(contoured, figsize = (5, 5))
print(coords)
```

```python
squares, _ = find_squares(result)
#print(squares)
contoured = cv2.drawContours(blank.copy(), squares, -1, (255, 0, 0), 1)
plot_image(contoured, figsize = (15, 15))
```

### Second Example

```python
biden = FlyDatasetReader("../data/2023-07-14_biden/", fly_glob = "*.JPG", max_size = 1_500)
biden1 = biden.read_fly(0)
plot_image(biden1)
```

```python
result = find_color_shapes(biden1)
squares, _ = find_squares(result)
#print(squares)
contoured = cv2.drawContours(biden1.copy(), squares, -1, (255, 0, 0), 1)
plot_image(contoured, figsize = (15, 15))
```

```python
result = find_color_shapes(biden1, min_h = 5, max_h = 20, kernel_size = 21)
#squares, _ = find_squares(result)
plot_image(result)
#print(squares)
#contoured = cv2.drawContours(biden1.copy(), squares, -1, (255, 0, 0), 1)
#plot_image(contoured, figsize = (15, 15))
```

## Template Matching

How well will the registration marks work with template matching?

```python
mark = blank[380:400, 258:278]
plot_image(mark, figsize = (2, 2))
```

```python
# Need to modify extract_match to find multiple matches.
loc = extract_match(blank, mark, cv2.TM_CCOEFF)

ax = plot_image(blank)
plot_box(loc, ax)
```

## Color Detection

Can we detect the marks by color?

```python

```
