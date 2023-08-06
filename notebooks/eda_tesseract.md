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

# Tesseract Experiments

This notebook contains experiments using Tesseract for optical character
recognition (OCR). We can use OCR to identify numbers on the rulers, which
makes it easier to locate the rulers and also to orient the image.


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
import pytesseract as pyt

from src.io import *
from src.match import *
from src.ops import *
from src.tsops import *
from src.viz import *
```

```python
biden = FlyDatasetReader("../data/2023-03-10_biden")#, max_size = 1_500)
blank = biden.read_blank()
img0 = biden.read(0)
```

```python
blank.shape
```

```python
shiny = FlyDatasetReader("../data/2023-03-27_shiny")#, max_size = 1_500)
s_blank = shiny.read_blank()
```

```python
plot_image_grid([blank, img0, s_blank], figsize = (15, 10))
```

## Preprocessing

```python
plot_image(blank, figsize = (10, 10))
```

```python
clahe = cv2.createCLAHE(clipLimit = 10)
transformed = cv2.cvtColor(simg0, cv2.COLOR_RGB2GRAY)
transformed = cv2.GaussianBlur(transformed, (0, 0), 1)
transformed = clahe.apply(transformed)
transformed = unsharp_mask(transformed, 1)
plot_image(transformed, figsize = (10, 10))
```

```python
#transformed = cv2.cvtColor(blank, cv2.COLOR_RGB2GRAY)
transformed = cv2.GaussianBlur(blank, (0, 0), .5)
transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2HSV)
transformed = cv2.inRange(transformed, (0, 0, 127), (255, 23, 255))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#transformed = cv2.erode(transformed, kernel)
plot_image(transformed, figsize = (10, 10))
```

## Tesseract

See the [DataLab OCR Workshop Reader][workshop-ocr]

[workshop-ocr]: https://ucdavisdatalab.github.io/workshop_ocr_python/chapters/01_ocr-basics.html

```python
pyt.image_to_string(transformed)
```

```python
pyt.image_to_osd(transformed, lang = "eng", config = r"digits")
```

```python
tsconfig = r"" #r"--psm 11 digits"
boxes = pyt.image_to_data(
    transformed, lang = "eng", output_type = "data.frame", config = tsconfig)
boxes.head()
```

```python
plot_ocr(transformed, boxes)
```

```python
def plot_ocr(img, boxes, figsize = (10, 10)):
    ax = plot_image(img, figsize = figsize)
    for i, row in boxes.iterrows():
        top, left, width, height = row[["top", "left", "width", "height"]]
        coords = (top, left, top + height, left + width)
        if row["conf"] > 0:
            #print(coords)
            plot_box(coords, ax)
    return ax
```

```python
boxes.query("conf > 0")
```

## Pipeline for Orienting Images

```python
oriented = orient_image(s_blank)
```

```python
plot_image(oriented)
```

```python
orient_image
```

```python
simg0 = shiny.read(0)
plot_image(simg0)
```

```python

```
