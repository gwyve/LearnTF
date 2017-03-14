import tensorflow as tf
from numpy import std
import os
from asyncore import read



from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('./MNIST_data',one_hot=True)

SKIP_STEP = 100


BATCH_SIZE = 50
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNEL = 1
CLASS_NUM = 10
LEARNING_RATE = 0.001
PRETRAIN_FILE = "./checkpoints/pretrain/"
TRAINED_FILE = "./checkpoints/trained/"
SUMMARY_PATH = "./summary/"

class mnist:
    
    def __init__(self,batch_size,image_width,image_height,image_channel,class_num,learning_rate,pretrain_file,trained_file,summary_path):
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.image_channel = image_channel
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.pretrain_file = pretrain_file
        self.trained_file = trained_file
        self.summary_path = summary_path
        self.global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name = "global_step")
    
    def __conv2d(self,x,W,b,name):
        return tf.nn.relu(tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME") + b, name)
    
    def __max_pool(self,x,name):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

    
    
    def _create_variable(self):
        with tf.name_scope("variable"):
            self.W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev = 0.1),name="W_conv1")
            self.b_conv1 = tf.Variable(tf.constant(0.1,shape=[32]),name="b_conv1")
            self.W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev = 0.1),name="W_conv2")
            self.b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]),name="b_conv2")
            self.W_fc1 = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev = 0.1),name = "W_fc1")
            self.b_fc1 = tf.Variable(tf.constant(0.1,shape=[1024]),name="b_fc1")
            self.W_fc2 = tf.Variable(tf.truncated_normal([1024,10],stddev = 0.1),name = "W_fc2")
            self.b_fc2 = tf.Variable(tf.constant(0.1,shape=[10],name="b_fc2"))
            
        
        
    def _create_placeholders(self):
        with tf.name_scope("data"):
            self.image_source = tf.placeholder(tf.float32, shape=[None,self.image_height *self.image_width * self.image_channel],name = "images")
            self.label = tf.placeholder(tf.float32, shape = [None,self.class_num],name="label")
            self.drop_out_prob = tf.placeholder(tf.float32, name="drop_out_prob")
            
            
    def _create_loss(self):
        with tf.device('/cpu:0'):
            with tf.name_scope("loss"):
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.predict),name="loss")
                
    def _create_optimizer(self):
        with tf.device('/cpu:0'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate,1e-4).minimize(self.loss,global_step = self.global_step)
            
    
    def _create_accuracy(self):
        with tf.device('/cpu:0'):
            with tf.name_scope("accuracy"):
                self.correct_prediction = tf.equal(tf.argmax(self.predict,1),tf.argmax(self.label,1),"correct_predict") 
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32),name="accuracy")
        
    def _create_model(self):
        with tf.device('/cpu:0'):
            with tf.name_scope("model"):
                self.image = tf.reshape(self.image_source,[-1,self.image_height,self.image_width,self.image_channel])
                self.h_conv1 = self.__conv2d(self.image, self.W_conv1,self.b_conv1,"conv1") 
                self.h_pool1 = self.__max_pool(self.h_conv1, "pool1")
                self.h_conv2 = self.__conv2d(self.h_pool1, self.W_conv2, self.b_conv2, "conv2")
                self.h_pool2 = self.__max_pool(self.h_conv2, "pool2")
                self.h_pool2_reshape = tf.reshape(self.h_pool2, [-1,7*7*64], "h_pool2_reshape")
                self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_reshape,self.W_fc1)+self.b_fc1, "h_fc1")
                self.h_fc_drop = tf.nn.dropout(self.h_fc1, keep_prob = self.drop_out_prob)
                self.predict = tf.matmul(self.h_fc_drop,self.W_fc2) + self.b_fc2
             
    def __create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram_loss", self.loss)
            self.summary_op = tf.summary.merge_all()     
            
    
    def build_graph(self):
        self._create_placeholders()
        self._create_variable()
        self._create_model()
        self._create_loss()
        self._create_accuracy()
        self._create_optimizer()
        self.__create_summaries()

    def train(self,num_train_steps,mnist):
        self.build_graph()
        reader= tf.train.Saver()
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.pretrain_file))
            if ckpt and ckpt.model_checkpoint_path:
                reader.restore(sess, ckpt.model_checkpoint_path)
                
            writer = tf.summary.FileWriter(self.summary_path +str(self.learning_rate),sess.graph)
            
            initial_step = self.global_step.eval();
            
            accuracy_feed_dict = {self.image_source:mnist_data.test.images,self.label:mnist_data.test.labels,self.drop_out_prob:1}
            
            print("initial_step:" +str(initial_step))
            for index in range(initial_step,initial_step + num_train_steps):
                image,label = mnist_data.train.next_batch(self.batch_size)
                
                feed_dict = {self.image_source:image,self.label:label,self.drop_out_prob:0.5}
                loss, _, summary = sess.run([self.loss,self.optimizer,self.summary_op],feed_dict = feed_dict)

                writer.add_summary(summary, global_step=index)

                if (index + 1) % SKIP_STEP == 0:
                    print("step "+ str(index + 1)+":" +str(sess.run(self.accuracy,feed_dict=accuracy_feed_dict)))
                    saver.save(sess, self.trained_file, self.global_step)
                   
            print("final:############" +sess.run(self.accuracy,feed_dict={self.image_source:mnist_data.test.images,self.label:mnist_data.test.labels,self.drop_out_prob:1}))  
            

model = mnist(BATCH_SIZE,IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNEL,CLASS_NUM,LEARNING_RATE,PRETRAIN_FILE,TRAINED_FILE,SUMMARY_PATH) 
model.train(20000, mnist)