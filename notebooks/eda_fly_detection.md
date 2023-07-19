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

# Fly Detection

This notebook contains experiments with using template matching for fly detection.


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
from src.tsops import *
from src.viz import *
```

```python
biden = FlyDatasetReader("../data/2023-03-10_biden")#, max_size = 1_500)
blank = biden.read_blank()
img0 = biden.read_fly(0)
```

```python
blank.shape
```

```python
shiny = FlyDatasetReader("../data/2023-03-27_shiny")#, max_size = 1_500)
s_blank = shiny.read_blank()
```

```python
plot_image_grid([blank, img0, s_blank], figsize = (15, 10)) #hello
```
