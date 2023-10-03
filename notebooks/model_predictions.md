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
from ultralytics import YOLO

from src.io import *
from src.viz import plot_image, plot_box
```

```python
model = YOLO("../models/fly-model/weights/best.pt")
```

```python
model.export(format = "onnx", imgsz = (384, 640))
```

```python
model_last = YOLO("../models/fly-model/weights/last.pt")
model_last.export(format = "onnx", imgsz = (384, 640))
```

```python
data = FlyDatasetReader("../outputs/2023-08-11_shiny/")
len(data)
```

```python
for path, img in data.iter_read():
    print(path)
    out_path = list(Path(path).parts)
    out_path[-2] = "predictions"
    out_path = Path(*out_path)

    out_path.parent.mkdir(exist_ok = True)

    result = model.predict(source = cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    result = result[0].boxes.data.cpu()
    boxes = result.numpy()
    print(boxes)
    break
    #boxes.shape

    fig, ax = plt.subplots(1, 1, figsize = (20, 20))
    plot_image(img, ax = ax)
    count = 0
    for i in range(boxes.shape[0]):
        if boxes[i, 4] < 0.5:
            count += 1
            continue
        plot_box(boxes[i, (1, 0, 3, 2)], ax)

    print(f"Skipped {count} boxes.")
    
    fig.savefig(out_path)
    print(f"Wrote '{out_path}'\n")
    plt.close(fig)
```

```python

```
