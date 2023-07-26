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
from src.ops import *
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
def rotate_image(image, angle):
    """Rotate an image by an arbitrary angle.
    """
    # Based on https://stackoverflow.com/a/9042907/1166039
    # Get center of image.
    dims = image.shape[1::-1] # cols, rows
    center = np.array(dims) / 2
    # Third rotation matrix argument is scale.
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation, dims, flags = cv2.INTER_LINEAR)

foo = rotate_image(img0, 45)
plot_image(foo)
```

```python
img0.shape[1::-1]
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
rotations += [rotate_image(template, a)[100:150, 100:150] for a in angles]
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
    matches = []
    for i in range(n_max):
        # Find location of current best match.
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(similarity)
        # Stop if best match's similarity is below threshold.
        if max_val < thresh:
            break
        # Otherwise, add the location to the list.
        matches.append((max_loc, max_val))
        # Change the values in the matching location's box to -1.
        # NOTE: We might want to leave edges intact in case flies are standing
        # close together.
        x = np.rint(max_loc[0] + w_shift).astype(int)
        y = np.rint(max_loc[1] + h_shift).astype(int)
        similarity[slice(*y), slice(*x)] = -1

    return matches
```

```python
matches = [x for r in rotations for x in extract_matches(img0a, r)]
len(matches)
```

```python
rotations[0].shape
```

```python
ax = plot_image(img0a, figsize = (15, 15))

for m, _ in matches:
    m = m[::-1]
    dims = np.array(rotations[0].shape[:2])
    coords = np.concatenate([m, m + dims])
    plot_box(coords, ax)
```

```python
matches16 = [x for r in rotations for x in extract_matches(img16a, r)]
len(matches)
```

```python
ax16 = plot_image(img16a, figsize = (15, 15))

for m, _ in matches16:
    m = m[::-1]
    dims = np.array(rotations[0].shape[:2])
    coords = np.concatenate([m, m + dims])
    plot_box(coords, ax16)
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
matches0_n = [x for r in rotations for x in extract_matches(img0a_n, r, 0.80)]
len(matches0_n)
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
