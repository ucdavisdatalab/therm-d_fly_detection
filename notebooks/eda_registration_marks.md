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
from src.registration import *
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
plot_image(blank)
```

```python
biden = FlyDatasetReader("../data/2023-07-14_biden/")#, max_size = 1_500)
biden1 = biden.read(0)
plot_image(biden1)
```

## Color Thresholding

Hue-saturation-value (HSV) typically records hue as angles along a color wheel,
from 0 to 360 degrees. 0 is pure red, 120 is pure green, and 240 is pure blue.
In OpenCV, hue ranges from 0 to 179, which appears to just be a rescaling (to
fit within 8 bits).

```python
# Find green pixels.
#image, min_h = 60, max_h = 120, kernel_size = 5
#mask = cv2.inRange(image_hsv, (min_h, 30, 127), (max_h, 255, 255))
#mask = cv2.inRange(image_hsv, (min_h, 127, 127), (max_h, 255, 255))

masked = mask_hsv(biden1, (60, 50, 127), (120, 255, 255), close_kernel = 11)
plot_image(masked, figsize = (10, 10))
```

```python
squares, _ = compute_squares(masked, 3)
contoured = cv2.drawContours(biden1.copy(), squares, -1, (0, 255, 0), 2)
plot_image(contoured, figsize = (15, 15))
```

```python
masked_orange = mask_hsv(biden1, (5, 50, 127), (20, 255, 255), close_kernel = 11)
#plot_image(masked_orange)

squares_orange, _ = compute_squares(masked_orange, 1)
contoured = cv2.drawContours(biden1.copy(), squares_orange, -1, (255, 0, 0), 2)
plot_image(contoured, figsize = (15, 15))
```

Given 3 green contours and 1 orange contour, determine the orientation of the
image and the arena bounding box.

In a correctly oriented image, the orange square should be at the top left
corner.

```python
def orient_and_bound(green, orange):
    # Compute median center of each square and thereby determine which the
    # orange square's corner.
    contours = [orange] + green
    m = np.stack([np.median(c, axis = 0) for c in contours])
    mark_corners = arg_corner_sort(m)

    match np.where(mark_corners == 0)[0]:
        case 0: # in tl
            orient = None
        case 1: # in tr
            orient = cv2.ROTATE_90_COUNTERCLOCKWISE
        case 2: # in br
            orient = cv2.ROTATE_180
        case 3: # in bl
            orient = cv2.ROTATE_90_CLOCKWISE

    # Next, find the bound of the arena.
    # Strategy: get top-left corner of top-left contour, etc...
    bound = np.empty_like(contours[0])
    for i, ix in enumerate(mark_corners):
        contour = contours[ix]
        corners = arg_corner_sort(contour)
        bound[i, :] = contour[corners[i]]
    
    return contours, mark_corners, orient, bound

z = orient_and_bound(squares, squares_orange[0])
z
```

```python
bound = z[3]
bound
contoured = cv2.drawContours(biden1.copy(), [bound], -1, (255, 0, 0), 2)
plot_image(contoured, figsize = (15, 15))
```

```python
cv2.boundingRect(bound)
```

Perspective Correction

Compute distances of bounding box sides and use that as the new image size.
Then make a perspective correction with `cv2.warpPerspective`.

```python

```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## Template Matching

How well will the registration marks work with template matching?
<!-- #endregion -->

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

```python

```
