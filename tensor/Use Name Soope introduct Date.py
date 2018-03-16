import tensorflow as tf
with tf.name_scope("ScopeA"):
    a = tf.add(2,4,name="A_add")
    b = tf.multiply(a,7,name="B_add")
with tf.name_scope("ScopeB"):
    c = tf.add(7,5,name="C_name")
    d=tf.multiply(c,4,name="D_name")
e=tf.add(b,d,name="ADDe")
writer=tf.summary.FileWriter('D://tensorboard/name_scope',graph=tf.get_default_graph())


