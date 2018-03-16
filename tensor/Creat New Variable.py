import tensorflow as tf
MyVar=tf.Variable(3,name="myVariable")
ZeroMatrix =tf.zeros([9])
OneVector=tf.ones([5])
Uniform=tf.random_uniform([4,4,4],minval=0,maxval=12)
Normal=tf.random_normal([4,4,4],mean=0,stddev=2.0)
sess=tf.Session()
print(sess.run(Uniform))
