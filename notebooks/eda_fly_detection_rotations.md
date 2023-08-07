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
import src.match as match
import src.ops as ops
from src.viz import *
```

```python
biden = FlyDatasetReader("../data/2023-03-10_biden")#, max_size = 1_500)
blank = biden.read_blank()
img0 = biden.read(0)
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
template_path = Path("../output/fly_template.npz")
if not template_path.is_file():
    np.savez_compressed(template_path, *rotations)
```

```python
img0a = img0[1470:3750, 1070:5350]
plot_image(img0a)
```

```python
img16 = biden.read(16)
img16a = img16[1470:3750, 1070:5350]
plot_image(img16a)
```

### SQDIFF_NORMED

```python
locs, scores = match.concatenate(
    [match.extract(
        img0a, r, n_max = 50, threshold = 0.9999, adaptive = True
        , verbose = True, metric = cv2.TM_SQDIFF_NORMED)
     for r in rotations])
len(scores)
```

```python
ax = plot_image(img0a, figsize = (15, 15))

for i in range(locs.shape[0]):
    plot_box(locs[i, :], ax)
```

### CCORR_NORMED

```python
locs, scores = match.concatenate(
    [match.extract(
        img0a, r, n_max = 100, threshold = 0.96
        , verbose = True, metric = cv2.TM_CCORR_NORMED)
     for r in rotations])
len(scores)
```

```python
ax = plot_image(img0a, figsize = (15, 15))

for i in range(locs.shape[0]):
    plot_box(locs[i, :], ax)
```

### CCOEFF_NORMED

```python
locs, scores = match.concatenate(
    [match.extract(img0a, r, verbose = True) for r in rotations])
len(scores)
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
kept = match.suppress_nonmax(locs, scores)
```

```python
ax = plot_image(img0a, figsize = (15, 15))

for i in range(kept.shape[0]):
    plot_box(kept[i, :], ax)
```

### More Examples

```python
locs16, scores16 = match.concatenate(
    [match.extract(img16a, r) for r in rotations])
len(scores16)
```

```python
ax16 = plot_image(img16a, figsize = (15, 15))

for i in range(locs16.shape[0]):
    plot_box(locs16[i, :], ax16)
```

```python
biden_new = FlyDatasetReader("../data/2023-07-14_biden/", fly_glob = "*.JPG")
img0_n = biden_new.read(0)
(1148, 1434, 4219, 2338)
img0a_n = img0_n[1434:1434 + 2338, 1148:1148 + 4219]
plot_image(img0a_n)
```

```python
locs0_n, scores0_n = match.concatenate(
    [match.extract(img0a_n, r, threshold = 0.9999,
                   adaptive = True) for r in rotations])
len(scores0_n)
```

```python
ax0_n = plot_image(img0a_n, figsize = (15, 15))

for i in range(locs0_n.shape[0]):
    plot_box(locs0_n[i, :], ax0_n)
```

## Thresholding
