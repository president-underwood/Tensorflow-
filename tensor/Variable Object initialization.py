'''import tensorflow as tf
Init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(Init)
import tensorflow as tf
Var1=tf.Variable(7,name="initialize_me")
init=tf.initialize_variables([Var1],name="inputVar")
sess=tf.Session()
import tensorflow as tf
my_var=tf.Variable(1)
My_var_timesTwo=my_var.assign(my_var*2)
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)
print(sess.run(My_var_timesTwo))
print(sess.run(My_var_timesTwo))'''