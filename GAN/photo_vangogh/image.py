import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import sys

model = tf.keras.models.load_model("monet_gen_g.keras", compile=False)

img_filepath = "test_images/test1.jpg"

input_img = tf.io.read_file(img_filepath)
input_img = tf.io.decode_jpeg(input_img, 3)
input_img = tf.cast(input_img, tf.float32)
input_img = (input_img - 127.5) / 127.5
input_img = tf.image.resize(input_img, (256, 256))
input_img = tf.expand_dims(input_img, axis=0)
output_img = model(input_img, training=True) 
output_img = tf.squeeze(output_img, axis=0)
output_img = output_img * 127.5 + 127.5
output_img = tf.cast(output_img, tf.uint8)
output_img = cv2.cvtColor(output_img.numpy(), cv2.COLOR_RGB2BGR)

input_cv = cv2.imread(img_filepath)
cv2.imshow("Input", input_cv)
cv2.imshow("Ouput", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
