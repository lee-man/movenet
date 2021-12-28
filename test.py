import numpy as np
sf = 0.4
# return a sample from the 'standard normal' distribution
# just a random value
a = np.random.randn()*0.4
scale = 256*np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
angle = 90*np.pi/180
print(np.sin(angle))
# -0.04096877699597984
# 1.1

# 0.1184911087646951
# 0.9107369420962519