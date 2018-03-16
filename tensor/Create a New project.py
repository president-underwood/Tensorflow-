import tensorflow as tf

graph=tf.Graph()
with graph.as_default():
    with tf.name_scope("variables"):
        Global_step=tf.Variable(0,dtype=tf.int64,trainable=False,name="Global_Step")#计算总共运行次数
        Global_output=tf.Variable(0.0,dtype=tf.float64,trainable=False,name="Global_Output")


    with tf.name_scope("transformation"):
        with tf.name_scope("input"):
            A=tf.placeholder(shape=[None],dtype=tf.float64,name="Input_A")
        with tf.name_scope("intermidiate"):
            B=tf.reduce_prod(A,name="Prod_B")
            C=tf.reduce_sum(A,name="Sum_B")
        with tf.name_scope("Out_Put"):
            Output=tf.add(B,C,name="output")

    with tf.name_scope("update"):
        Update_total=Global_output.assign_add(Output)
        increment_step=Global_step.assign_add(1)
    with tf.name_scope("summary"):
            AVG=tf.div(Update_total,tf.cast(increment_step,tf.float64),name="Average")
            #汇总总数据
            tf.summary.scalar("Output_summary",Output)
            tf.summary.scalar("Total_summary",Update_total)
            tf.summary.scalar("avg_summary",AVG)
    with tf.name_scope("Global_ops"):
            Init=tf.initialize_all_variables()
            merged_summary=tf.summary.merge_all()

    sess=tf.Session(graph=graph)
    writer=tf.summary.FileWriter('D://tensorboard/DateFlowDiagram',graph)
    sess.run(Init)
    def run_Graph(Input_tensor):
        feed_dict={A:Input_tensor}
        _,step,summary=sess.run([Output,increment_step,merged_summary],feed_dict=feed_dict)
        writer.add_summary(summary,global_step=step)
run_Graph([7,15,35])
run_Graph([6,5,17])
run_Graph([25,19,34])
run_Graph([17,5])
run_Graph([17,5,3,9])
run_Graph([5,9,15,1,16])

writer.flush()