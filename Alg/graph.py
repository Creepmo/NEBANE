#coding:utf-8
import numpy as np
import networkx as nx
import random
from sklearn.model_selection import train_test_split

# 输入文件
# node_attr 节点属性文件，第一行为节点个数，下面每一行依次为从node id0开始的每个节点的属性，每个属性以空格分隔
# link_attr 边属性文件，每一行格式为:head_id tail_id#link attr(每个属性以空格分隔)
# label 节点label文件，格式为node_id node_name node_label
# init/ 节点embedding初始化文件，以deepwalk pre-train产生的embedding作为初始化embedding
class Graph(object):
    
    def __init__(self,args):
        
        self.nodeattrpath = args.datapath+"/node_attr"
        self.linkattrpath = args.datapath+"/link_attr"
        self.train_size = args.train_size
        
        self.read_files()
        
        if(self.train_size == 1.0):
            self.links_train = self.links
            self.links_attr_train = self.links_attr
        else:
            self.link_split()

    # 读入node和link属性文件
    def read_files(self):
        nodeattrfile = open(self.nodeattrpath,'r')
        linkattrfile = open(self.linkattrpath,'r')

        self.nodes_attr = []
        self.nodes_num = int(nodeattrfile.readline().strip())
        for line in nodeattrfile.readlines():
            self.nodes_attr.append(np.array([float(i) for i in line.strip().split(" ")]))
        
        self.links = []
        self.links_attr = []
        for line in linkattrfile.readlines():
            item = line.strip().split("#")
            link = tuple([int(i) for i in item[0].split(" ")])
            self.links.append(link)
            self.links_attr.append(np.array([float(i) for i in item[1].split(" ")]))
        self.link_attr_size = len(self.links_attr[0])

    # 将link分为训练集和测试集
    def link_split(self):
    
        self.links_train,self.links_test,self.links_attr_train,self.links_attr_test \
        = train_test_split(self.links,self.links_attr, random_state = 1, \
            test_size = 1.0 - self.train_size)

        linktestfile = open(self.linkattrpath+"_test"+str(np.round((1.0-self.train_size)*100)),"w")

        for i,link in enumerate(self.links_test):
            linktestfile.write("%d %d#"%(link[0],link[1]))
            for attr in self.links_attr_test[i]:
                linktestfile.write(" %d"%(attr))
            linktestfile.write("\n")
        linktestfile.close()

        linktrainfile = open(self.linkattrpath+"_train"+str(np.round((1.0-self.train_size)*100)),"w")

        for i,link in enumerate(self.links_train):
            linktrainfile.write("%d %d#"%(link[0],link[1]))
            for attr in self.links_attr_train[i]:
                linktrainfile.write(" %d"%(attr))
            linktrainfile.write("\n")
        linktrainfile.close()
        

