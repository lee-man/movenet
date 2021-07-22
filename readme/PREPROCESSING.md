# Data and Target Preprocessing and Preparation

## Object detection
Two related files: [coco.py](../src/lib/datasets/dataset/coco.py) and [ctdet.py](../src/lib/datasets/sample/ctdet.py).

**Coordination definition of image space**: x - vertical axis, width; y - horizontal axis, height.
### The pipeline of image inputs
1. Raw image file -> (cv.imread) -> np.ndarray with shape of (height, width, channel)
2. -> (random flip, random scaling, cropping and color jittering)
3. -> (/255) -> (- mean, / std) -> (transpose 3, 0, 1) -> normalized np.ndarray with shape of (channel, height, width)
4. 

### The pipeline of annotations

1. Raw annotation -> bbox: [x1, y1, x2, y2] in range of raw image size
2. -> (calculate the center point) -> ct [mid_x, mid_y]
3. 