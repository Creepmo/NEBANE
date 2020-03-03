#coding:utf-8
import argparse
from graph import Graph
from train import Train
from evaluation import Eval
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Run NEBANE.")
    parser.add_argument("--datapath",nargs='?',default="./Datasets/Amazon",help="AMiner Amazon") #AMiner Amazon
    parser.add_argument("--embdim",type=int,default=256,help="the dimension of embeddings")
    parser.add_argument("--batch_size",type=int,default=200,help="batch size")
    parser.add_argument("--epoch_num",type=int,default=20,help="the number of epoches")
    parser.add_argument("--num_sampled",type=int,default=5,help="the number of negative samples")
    parser.add_argument("--eta",type=float,default=1.0,help="the weight of regularization loss")
    parser.add_argument("--alpha",type=float,default=1000,help="the weight of node-node loss")
    parser.add_argument("--beta",type=float,default=1000,help="the weight of link-node loss")
    parser.add_argument("--lr",type=float,default=0.01,help="the learning rate")
    parser.add_argument("--train_size",type=float,default=1.0,help="the training ratio")

    return parser.parse_args()

def main(args):
    
    for args.train_size in [1.0,0.9,0.4,0.6,0.8]:
        graph = Graph(args)
        train = Train(args,graph)
        evaluation = Eval(args)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)