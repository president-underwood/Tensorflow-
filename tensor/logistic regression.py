import tensorflow as tf
import os

W = tf.Variable(tf.zeros([5,1]),name="w")
b = tf.Variable(0.,name="b")
def combine_inputs(X):
    return tf.matmul(X, W) + b
def inference(X):
    return tf.sigmoid(combine_inputs(X))
def loss(X, Y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=combine_inputs(X),logits=Y))#通过交叉熵求误差
def Read_Csv(Batch_Size,File_Name,Record_Deults):
    file_name_queue=tf.train.string_input_producer([os.path.join(os.getcwd(),File_Name)])
    reader=tf.TextLineReader(skip_header_lines=1)
    key,value=reader.read(file_name_queue)
    decoded=tf.decode_csv(value,record_defaults=Record_Deults)#通过使用decode_csv函数将文本行转换到具有指定默认值的由张量列组成的元组中。
    return tf.train.shuffle_batch(decoded,batch_size=Batch_Size,capacity=Batch_Size*50,min_after_dequeue=Batch_Size)
def inputs():
    passenger_id,survived,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked= \
        Read_Csv(100, "train.csv", [[0.0], [0.0], [0], [""], [""], [0.0], [0.0], [0.0], [""], [0.0], [""], [""]])
    # convert categorical data
    is_first_class = tf.to_float(tf.equal(pclass, [1]))#设置三个变量对应一二三等座
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))
    gender = tf.to_float(tf.equal(sex, ["female"]))#像性别这种用一个变量表示就可以了，eg.female is 1 but male is 0
    # 最后我们将所有特征放入一个单独的矩阵当中
    # 之后我们将矩阵转置，使得每一行对应一个样本，每一列对应一个特征， transpose to have a matrix with one example per row and one feature per column.
    features = tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, age]))
    survived = tf.reshape(survived, [100, 1])
    return features, survived
def train(total_loss):
    learning_rate=0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
def evaluate(sess,X,Y):
    predicted=tf.cast(inference(X)>0.5,tf.float32)
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted,Y),tf.float32))))
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    X,Y=inputs()
    total_loss=loss(X,Y)
    Train_Op=train(total_loss)
    cooder=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=cooder)
    train_steps=1000
    for step in  range(train_steps):
        sess.run([Train_Op])
        if step % 10==0:
            print("loss:",sess.run([total_loss]))

    evaluate(sess,X,Y)
    import time
    time.sleep(5)
    cooder.request_stop()
    cooder.join(threads)
