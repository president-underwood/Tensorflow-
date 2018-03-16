import tensorflow as tf
import numpy as np
a=tf.placeholder(tf.int64,shape=[2],name="MyInput")
b=tf.reduce_prod(a,name="prodB")
c=tf.reduce_sum(a,name="sumC")
d=tf.add(b,c,name="inputD")
sess=tf.Session()
input_dict={a:np.array([4,7],dtype=np.int64)}
print(sess.run(d,feed_dict=input_dict))

