import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio


interpreter = tf.lite.Interpreter(model_path="../../models/lite-model_movenet_singlepose_lightning_3.tflite")
interpreter.allocate_tensors()

'''
Check input/output details
'''
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("== Input details ==")
print("name:", input_details[0]['name'])
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])
print("\n== Output details ==")
print("name:", output_details[0]['name'])
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])


'''
This gives a list of dictionaries. 
'''
tensor_details = interpreter.get_tensor_details()

for dict in tensor_details:
    i = dict['index']
    tensor_name = dict['name']
    shape = dict['shape']
    # scales = dict['quantization_parameters']['scales']
    # zero_points = dict['quantization_parameters']['zero_points']
    # tensor = interpreter.tensor(i)()

    print(i, type, tensor_name, shape)# , scales.shape, zero_points.shape, tensor.shape)
