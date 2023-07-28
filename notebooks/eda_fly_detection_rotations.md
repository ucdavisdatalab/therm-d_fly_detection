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

# Rotation Experiments for Fly Detection

This notebook contains experiments with using template matching for fly
detection, with functions to aid rotation.


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
import src.ops as ops
#from src.tsops import *
from src.viz import *
```

```python
biden = FlyDatasetReader("../data/2023-03-10_biden")#, max_size = 1_500)
blank = biden.read_blank()
img0 = biden.read_fly(0)
```

## Rotation

```python
foo = ops.rotate(img0, 45)
plot_image(foo)
```

```python
#clahe = cv2.createCLAHE(clipLimit = 10)
#img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
#img0 = cv2.GaussianBlur(img0, (0, 0), 1)
#img0 = clahe.apply(img0)
#plot_image(img0)
```

Now let's look at a specific fly template.

```python
template = img0[1975:2225, 1402:1652]
print(template.shape)
ax = plot_image(template, grayscale=True)
plot_box((124, 124, 125, 125), ax)
```

```python
angles = range(45, 360, 15)
rotations = [template[100:150, 100:150]]
rotations += [ops.rotate(template, a)[100:150, 100:150] for a in angles]
plot_image_grid(rotations)
```

```python
img0a = img0[1470:3750, 1070:5350]
plot_image(img0a)
```

```python
img16 = biden.read_fly(16)
img16a = img16[1470:3750, 1070:5350]
plot_image(img16a)
```

```python
def extract_matches(image, template, thresh = 0.85, n_max = 100):
    """Extract multiple matches of a template from an image.

    Arguments
    ---------
    image
        The image from which to extract matches.

    template
        The template to search for in the image.

    thresh
        The similarity threshold for matches.

    n_max
        The maximum number of matches to return.
    """
    similarity = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    h, w = template.shape[:2]
    w_shift = np.array([-w, w]) / 2
    h_shift = np.array([-h, h]) / 2
    # Print 99.9th percentile, median, and mean.
    print(f"{np.percentile(similarity, 99.9)} {np.median(similarity)} {np.mean(similarity)}")

    # Find up to n_max matches.
    #matches = []
    locs = np.zeros((n_max, 4), dtype = np.int32)
    vals = np.zeros(n_max)
    for i in range(n_max):
        # Find location of current best match.
        # Locations are top left corner of match.
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(similarity)
        # Stop if best match's similarity is below threshold.
        if max_val < thresh:
            i -= 1
            break
        # Otherwise, add the location to the list.
        vals[i] = max_val
        x, y = max_loc
        locs[i, :] = (y, x, y + h, x + w)
        # Change the values in the matching location's box to -1.
        # NOTE: We might want to leave edges intact in case flies are standing
        # close together.
        similarity[y:y + h, x:x + h] = -1

    return locs[:i + 1, :], vals[:i + 1]
```

```python
def concatenate_matches(matches):
    return [np.concatenate(x) for x in zip(*matches)]

locs, vals = concatenate_matches(
    [extract_matches(img0a, r) for r in rotations])
len(vals)
```

```python
rotations[0].shape
```

```python
ax = plot_image(img0a, figsize = (15, 15))

for i in range(locs.shape[0]):
    plot_box(locs[i, :], ax)
```

### Non-maximum Suppression

When several boxes have a large overlap, ignore all but the box with the
highest score.

```python
locs
```

```python
def fast_iou(a, b, area):
    # Find intersection area.
    h = min(a[2], b[2]) - max(a[0], b[0])
    w = min(a[3], b[3]) - max(a[1], b[1])
    int_area = max(0, h) * max(0, w)
    return int_area / (2 * area - int_area)

# See:
#    https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/
# Iterate on selecting the best match and removing high IoU boxes.
candidates = np.argsort(vals)  # smallest to largest
kept = np.zeros_like(locs)
n_boxes = 0
while len(candidates) > 0:
    # Find best match.
    i = candidates[-1]
    candidates = candidates[:-1]
    best = locs[i, :]
    kept[n_boxes] = best
    n_boxes += 1

    # Remove all boxes with high IoU compared to best match.
    c_locs = locs[candidates, :]
    h = np.fmin(best[2], c_locs[:, 2]) - np.fmax(best[0], c_locs[:, 0])
    w = np.fmin(best[3], c_locs[:, 3]) - np.fmax(best[1], c_locs[:, 1])
    int_area = np.fmax(0, h) * np.fmax(0, w)
    ious = int_area / (2 * 2500 - int_area)
    candidates = candidates[ious < 0.60]

kept = kept[:n_boxes]
```

```python
ax = plot_image(img0a, figsize = (15, 15))

for i in range(kept.shape[0]):
    plot_box(kept[i, :], ax)
```

### More Examples

```python
locs16, vals16 = concatenate_matches(
    [extract_matches(img16a, r) for r in rotations])
len(vals16)
```

```python
ax16 = plot_image(img16a, figsize = (15, 15))

for i in range(locs16.shape[0]):
    plot_box(locs16[i, :], ax16)
```

```python

```

```python
biden_new = FlyDatasetReader("../data/2023-07-14_biden/", fly_glob = "*.JPG")
img0_n = biden_new.read_fly(0)
(1148, 1434, 4219, 2338)
img0a_n = img0_n[1434:1434 + 2338, 1148:1148 + 4219]
plot_image(img0a_n)
```

```python
locs0_n, vals0_n = concatenate_matches(
    [extract_matches(img0a_n, r, 0.8) for r in rotations])
len(vals0_n)
```

```python
ax0_n = plot_image(img0a_n, figsize = (15, 15))

for i in range(locs0_n.shape[0]):
    plot_box(locs0_n[i, :], ax0_n)
```

```python
ax0_n = plot_image(img0a_n, figsize = (15, 15))

for m, _ in matches0_n:
    m = m[::-1]
    dims = np.array(rotations[0].shape[:2])
    coords = np.concatenate([m, m + dims])
    plot_box(coords, ax0_n)
```

## Thresholding
