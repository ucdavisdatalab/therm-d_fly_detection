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

```python
mark = blank[380:400, 258:278]
plot_image(mark, figsize = (2, 2))
```

## Template Matching

How well will the registration marks work with template matching?

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
