# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:51:47 2023

@author: tomne
"""

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import sys

model = tf.keras.models.load_model("p2p_gen_facades.keras", compile=False)


cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Unable to video capture")
    cap.release()
    sys.exit()
    
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Unable to retrieve frame")
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (256, 256))
    frame_rgb = (frame_rgb - 127.5) / 127.5
    input_img = tf.expand_dims(frame_rgb, axis=0)
    
    output_img = model(input_img, training=True)
    output_img = tf.squeeze(output_img, axis=0)
    output_img = output_img * 127.5 + 127.5
    output_img = tf.cast(output_img, tf.uint8)
    output_img = cv2.cvtColor(output_img.numpy(), cv2.COLOR_RGB2BGR)
    
    cv2.imshow("Input", frame)
    cv2.imshow("Output", output_img)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
