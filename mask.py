import tensorflow as tf
import  numpy as np

crop_features = tf.random_uniform((3,3,7),0,1)
mask = tf.random_uniform((3,3),0,1)
mask_out_1 = tf.expand_dims(tf.to_float(tf.greater(mask,0.6)),axis=-1)
mask_out_2 = tf.tile(mask_out_1,(1,1,crop_features.get_shape().as_list()[-1                                                                                  ]))
mask_in = tf.multiply(crop_features,mask_out_2)
sess = tf.Session()
out_2,out_3 = sess.run([crop_features,mask_in])
print(out_2,out_3)