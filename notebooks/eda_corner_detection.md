---
jupyter:
  jupytext:
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

# Corner Detection Experiments


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

## Corner Detection

```python
grayed = cv2.cvtColor(s_blank, cv2.COLOR_RGB2GRAY)
```

```python
help(cv2.cornerHarris)
```

```python
cornered = cv2.cornerHarris(grayed, 2, 3, 0.04)
cornered = cv2.dilate(cornered, None)
s_blank2 = s_blank.copy()
s_blank2[cornered > 0.005*cornered.max()] = [0, 255, 0]
plot_image(s_blank2, figsize = (15, 15))
```

```python
# Goal: get the coordinates of each point and try to determine which points lie
# along a line.
```

```python

```
