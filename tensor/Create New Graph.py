import tensorflow as tf
graph=tf.Graph()
with graph.as_default():
    in_1=tf.placeholder(tf.float64,shape=[],name="inputA")
    in_2=tf.placeholder(tf.float64,shape=[],name="inputB")
    const=tf.constant(5,dtype=tf.float64,name="static_value")
    with tf.name_scope("Transformation"):
        with tf.name_scope("A"):
            A_mul=tf.multiply(in_1,const)
            A_out=tf.subtract(A_mul,in_1)
        with tf.name_scope("B"):
            B_mul=tf.multiply(in_2,const)
            B_out=tf.subtract(B_mul,in_2)
        with tf.name_scope("C"):
            C_div=tf.div(A_out,B_out)
            C_out=tf.add(C_div,const)
        with tf.name_scope("D"):
            D_div=tf.div(B_out,A_out)
            D_out=tf.add(D_div,const)
    Out=tf.maximum(C_out,D_out)
    writer=tf.summary.FileWriter('D://tensorboard/name_scope_3',graph=graph)

