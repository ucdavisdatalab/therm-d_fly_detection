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

```python
plot_image(img0)
```

```python
arena0 = img0[1400:3750, 1100:5400]
plot_image(arena0)
```

Loop that generates all img and arenas 

```python
for i in range(0,23):
    locals()['img' + str(i)] = biden.read_fly(i)
    locals()['arena' + str(i)] = biden.read_fly(i)[1400:3750, 1100:5400]    
```

```python
template = arena0[2260:2290, 380:435]
plot_image(template)
```

```python
arena = arena5

loc = extract_match(arena, template, cv2.TM_CCOEFF_NORMED)
ax = plot_image(arena)
plot_box(loc, ax)
```

Draws rectangle on best match same as above

```python
import matplotlib.patches as patches

match = cv2.matchTemplate(arena, template, cv2.TM_CCOEFF_NORMED) #TM_CCOEFF and TM_CCOEFF_NORMED options!!!
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

threshold = 0.8  #this is for the intensitiy of the picture
if max_val >= threshold:
    h, w, _ = template.shape
    top_left = max_loc
    fig, ax = plt.subplots()
    ax.imshow(arena)
    rect = patches.Rectangle(top_left, .75*w, .75*h, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    ax.set_title('Target Image with Best Match')
    plt.show()
```

```python
match 
print(max_loc)
cv2.minMaxLoc(match)

```

```python
plot_image(arena)
```

This uses mario coins method to try to template match the flies

```python
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_gray = cv.cvtColor(arena, cv.COLOR_BGR2GRAY)
temp_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
h, w = temp_gray.shape

res = cv.matchTemplate(img_gray,temp_gray,cv.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
 cv.rectangle(arena, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
cv.imwrite('res.png',arena)
mat = cv.imread('res.png', cv.IMREAD_COLOR)

plt.imshow(mat)
plt.title("RGB Image")
plt.axis("off")  # Turn off axis
plt.show()

# Display the target image
#ax.imshow('res.png', cv.IMREAD_COLOR)
```

## K Best template matches

```python
target_image = arena
template_image = template
matching_result = cv2.matchTemplate(arena, template, cv2.TM_CCOEFF_NORMED)
h, w, channels = template_image.shape

k = 10 # Number of k flies
match_locations = [] #list to append location of matc
for i in range(k): #loops through number of flies
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matching_result) #looks at the location of match
    match_locations.append(max_loc) #adds that location to the list
    matching_result[max_loc[1] - round(.5*h) :max_loc[1] + round(.5*h), max_loc[0] - round(.5*w) :max_loc[0] + round(.5*w)] = -1 #change the values at the location to 0

fig, ax = plt.subplots()

ax.imshow(target_image)

for match_loc in match_locations:
    top_left = match_loc
    rect = patches.Rectangle(top_left, w, h, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

plt.show()
```

```python
round(.5*h)
```

```python

```

```python
target_image = arena
template_image = template
matching_result = cv2.matchTemplate(arena, template, cv2.TM_CCOEFF_NORMED)
template.shape
```

```python
matching_result[max_loc[1]:max_loc[1] + h, max_loc[0]:max_loc[0] + w]
```

```python
test = arena[max_loc[1]:max_loc[1] + h, max_loc[0]:max_loc[0] + w]

```

```python
print(match_locations[0])
match_locations[1]

matching_result[936,415]
```

```python
img_gray = cv.cvtColor(arena, cv.COLOR_BGR2GRAY)
temp_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
h, w = temp_gray.shape

res = cv.matchTemplate(img_gray,temp_gray,cv.TM_CCOEFF_NORMED)
```
