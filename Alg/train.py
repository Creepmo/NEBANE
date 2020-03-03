#coding:utf-8
import tensorflow as tf
import numpy as np
import math
from collections import Counter

class Train(object):
    def __init__(self,args,graph):

        # parameter settings and inputs
        self.embdim = int(args.embdim/2)
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch_num
        self.num_sampled = args.num_sampled

        self.eta = args.eta
        self.alpha = args.alpha
        self.beta = args.beta
        self.lr = args.lr
        self.train_size = args.train_size
        self.datapath = args.datapath

        self.embfile = open(self.datapath+"/node_emb","w")

        self.nodes_attr = np.array(graph.nodes_attr)
        self.links_attr = np.array(graph.links_attr_train)
        self.links = np.array(graph.links_train)

        self.node_attr_size = len(graph.nodes_attr[0])
        self.link_attr_size = graph.link_attr_size
        self.nodes_num = graph.nodes_num

        self.recordloss = args.recordloss
        if(self.recordloss):
            self.lossfile = open("loss.txt","a+")
            dataset = args.datapath.split("/")[-1]
            self.lossfile.write("%s:\n"%dataset)
            self.lossfile.write("Epoch\tLoss\tNode Loss\tLink Loss\n")
        
        self.init_model()
        self.run()

    # 初始化tensorflow模型图
    def init_model(self):

        with tf.device(":/cpu:0"):
                   
            self.h = tf.placeholder(tf.int32,[None],name="h")
            self.t = tf.placeholder(tf.int32,[None],name="t")
            self.link_attr = tf.placeholder(tf.float32,[None,self.link_attr_size],name="link_attr")
            self.nodes_attr_tf = tf.placeholder(tf.float32,[self.nodes_num,self.node_attr_size],name="node_attr")
            
            self.nodes_emb_id = tf.Variable(tf.random_uniform([self.nodes_num,self.embdim],-1.0,1.0))
            self.nodes_emb_out = tf.Variable(tf.truncated_normal([self.nodes_num,self.embdim*2],\
                stddev=6.0/math.sqrt(self.embdim*2)))
            # self.nodes_emb_out = tf.Variable(self.load_init_emb())
            self.softmax_biases = tf.zeros([self.nodes_num])

            self.t_flat = tf.reshape(self.t,[-1,1])
            self.h_flat = tf.reshape(self.h,[-1,1])
                        
            self.W_node = tf.Variable(tf.random_uniform([self.node_attr_size,self.embdim],-1.0,1.0,name="W_node"))
            self.nodes_emb_attr = tf.nn.tanh(tf.matmul(self.nodes_attr_tf,self.W_node))

            self.nodes_emb_in = tf.concat([self.nodes_emb_id,self.eta * self.nodes_emb_attr],axis=1)
            
            self.h_emb = tf.nn.embedding_lookup(self.nodes_emb_in,self.h)
            self.t_emb = tf.nn.embedding_lookup(self.nodes_emb_in,self.t)

            self.W_link = tf.Variable(tf.truncated_normal([self.link_attr_size,self.embdim*2],stddev=0.3),name="W_link")
            self.link_emb = tf.nn.tanh(tf.matmul(self.link_attr,self.W_link))

            self.node_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.nodes_emb_out,self.softmax_biases,\
                self.t_flat,self.h_emb,self.num_sampled,self.nodes_num))
            self.node_loss += tf.reduce_mean(tf.nn.sampled_softmax_loss(self.nodes_emb_out,self.softmax_biases,\
                self.h_flat,self.t_emb,self.num_sampled,self.nodes_num))

            self.link_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.nodes_emb_out,self.softmax_biases,\
                self.h_flat,self.link_emb,self.num_sampled,self.nodes_num))
            self.link_loss += tf.reduce_mean(tf.nn.sampled_softmax_loss(self.nodes_emb_out,self.softmax_biases,\
                self.t_flat,self.link_emb,self.num_sampled,self.nodes_num))
            self.link_loss += tf.reduce_mean(tf.nn.sampled_softmax_loss(self.nodes_emb_in,self.softmax_biases,\
                self.h_flat,self.link_emb,self.num_sampled,self.nodes_num))
            self.link_loss += tf.reduce_mean(tf.nn.sampled_softmax_loss(self.nodes_emb_in,self.softmax_biases,\
                self.t_flat,self.link_emb,self.num_sampled,self.nodes_num))

            self.reg_loss = tf.nn.l2_loss(self.W_node) + tf.nn.l2_loss(self.W_link)

            self.loss = self.alpha * self.node_loss + self.beta * self.link_loss + self.reg_loss

            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            # self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

    # 读入初始化embedding，以deepwalk训练得到的embedding pre-train
    def load_init_emb(self):

        # self.initembfile_all = open(args.datapath+"init/node_emb","r")
        if(self.train_size == 1.0):
            initembfile = open(self.datapath+"/init/node_emb","r")
        else:
            initembfile = open(self.datapath+"/init/node_emb"+"_train"+str(np.round((1.0-self.train_size)*100)),"r")

        initembfile.readline()
        node_embeddings = np.array(np.random.normal(-6.0/math.sqrt(self.embdim*2),6.0/math.sqrt(self.embdim*2),\
            (self.nodes_num,self.embdim*2))).astype(np.float32)
        for line in initembfile.readlines():
            item = line.strip().split()
            node_embeddings[int(item[0])] = np.array([float(i) for i in item[1:]])

        return node_embeddings
    
    # 训练模型函数
    def run(self):

        with tf.device(":/cpu:0"):

            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                for epoch in range(self.epoch_num):
                    batches = self.gen_batch()
                    loss = 0.0
                    link_loss = 0.0
                    node_loss = 0.0
                    for batch_id,batch in enumerate(batches):
                        h_batch,t_batch,link_attr_batch = batch
                        feed_dict = {self.h:h_batch,self.t:t_batch,self.link_attr:link_attr_batch,\
                        self.nodes_attr_tf:self.nodes_attr}
                        _,loss_batch,link_loss_batch,node_loss_batch = sess.run([self.optimizer,self.loss,\
                            self.link_loss,self.node_loss],feed_dict=feed_dict)
                        loss += loss_batch
                        link_loss += link_loss_batch
                        node_loss += node_loss_batch
                        
                    print("epoch %d,loss = %.4f,link_loss = %.4f,node_loss = %.4f"%(epoch+1,loss,link_loss,node_loss))
                    if(self.recordloss):
                        self.lossfile.write("%d\t%.4f\t%.4f\t%.4f\n"%(epoch+1,loss,link_loss,node_loss))
                
                nodes_emb_in = sess.run(self.nodes_emb_in,feed_dict={self.nodes_attr_tf:self.nodes_attr})
                nodes_emb_out = sess.run(self.nodes_emb_out)

                nodes_emb = nodes_emb_in + nodes_emb_out
                        
            self.embfile.write("%d %d\n"%(self.nodes_num,self.embdim*2))
            for i in range(self.nodes_num):
                self.embfile.write("%d "%i)
                for j in range(self.embdim*2):
                    self.embfile.write("%f "%nodes_emb[i][j])
                self.embfile.write("\n")
            self.embfile.close()

    # 产生batch函数
    def gen_batch(self):

        input_data = self.links
        data_size = len(input_data)
        indices = np.random.permutation(np.arange(data_size))
        start_index = 0
        end_index = min(start_index + self.batch_size,data_size)
        while(start_index < data_size):
            index = indices[start_index:end_index]
            h_batch = input_data[index][:,0]
            t_batch = input_data[index][:,1]
        
            link_attr_batch = self.links_attr[index]

            yield h_batch,t_batch,link_attr_batch
            start_index = end_index
            end_index = min(start_index + self.batch_size,data_size)    