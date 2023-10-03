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

```python
# Allow imports from `..`
import os
import sys

module_path = os.path.abspath("..")
if module_path not in sys.path:
    sys.path.append(module_path)
```

```python
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
#from ultralytics import YOLO
import onnxruntime

from src.io import *
from src.viz import plot_image, plot_box
```

```python
data = FlyDatasetReader("../outputs/2023-08-11_shiny/")
len(data)
```

```python
ort_session = onnxruntime.InferenceSession("../models/fly-model/weights/last.onnx")
```

```python
ort_session.get_inputs()[0].name
```

```python
# Resize image to 640x640.
path, img = data.read(0)
print(path)

img = rescale(img, 640)
img = cv2.copyMakeBorder(
    img, 0, 640 - img.shape[0], 0, 640 - img.shape[1], cv2.BORDER_CONSTANT, 0)
orig = img.copy()

plot_image(img)

# Convert image to float32 in BGR.
#img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img = img.astype(np.float32)
img /= 255. #img.max()

# Move channels to first dimension.
img = np.moveaxis(img, -1, 0)
print(img.shape)

ort_inputs = {"images": [img]}
```

```python
ort_outputs = ort_session.run(None, ort_inputs)
```

```python
ort_outputs[0].shape
```

```python
ort_outputs[0][0, :, :].transpose()
```

```python
is_nonzero = ort_outputs[0][0, 4, :] > 0.001
boxes = ort_outputs[0][0, :, np.where(is_nonzero)]
boxes = np.squeeze(boxes)
boxes.shape
```

```python
fig, ax = plt.subplots(1, 1, figsize = (20, 20))
plot_image(orig, ax = ax)
count = 0
for i in range(boxes.shape[0]):
    if boxes[i, 4] < 0.05:
        count += 1
        continue
    #plot_box(boxes[i, (1, 0, 3, 2)], ax)
    xc, yc, w, h = boxes[i, :4]
    plot_box((yc - h/2, xc - w/2, yc + h/2, xc + w/2), ax)

print(f"Skipped {count} boxes.")
```

```python

```
