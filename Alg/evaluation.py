#coding:utf-8
from sklearn.model_selection import train_test_split
from sklearn import svm,metrics
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

class Eval(object):
	def __init__(self,args):
		
		self.resultfile = open("result.txt","a+")
		self.train_size = args.train_size
		self.embfilepath = args.datapath+"/node_emb"
		self.labelfilepath = args.datapath+"/label"
		self.linkattrpath = args.datapath+"/link_attr"
		dataset = args.datapath.split("/")[-1]
		self.read_embfile(self.embfilepath)
		self.read_labelfile(self.labelfilepath)
		
		if self.train_size == 1.0:
			self.resultfile.write("dataset= %s,alpha= %.2f,beta= %.2f,eta= %.2f\n"%(dataset,\
			args.alpha,args.beta,args.eta))
			self.node_classfication()
			self.node_clustering()
			# self.network_visualization()
			self.similarity_search()
		else:
			if(self.train_size < 0.9):
				if(self.train_size == 0.4):
					self.resultfile.write("\tlink prediction:\n\t\tauc: ")
				self.link_prediction(self.linkattrpath)
			else:
				self.node_recommendation(self.linkattrpath)

	# 读入embedding文件
	def read_embfile(self,embfilepath):

		embfile = open(embfilepath,"r")
		item = embfile.readline().strip().split()
		self.nodes_num_total = int(item[0])
		self.embdim = int(item[1])
		self.nodes_emb = []
		for line in embfile.readlines():
			item = line.strip().split()
			node_emb = item[1:]
			self.nodes_emb.append([float(i) for i in node_emb])

	# 读入label文件
	def read_labelfile(self,labelfilepath):

		labelfile = open(labelfilepath,"r")
		self.nodes_label = []
		for line in labelfile.readlines():
			item = line.strip().split()
			self.nodes_label.append(int(item[2]))
		self.nodes_num = len(self.nodes_label)

	# 读入link文件
	def read_edgefile(self,edgefilepath):

		edgefile = open(edgefilepath,"r")
		nodes_neibor = {}
		for line in edgefile.readlines():
			item = line.strip().split("#")
			pos_edge = [int(i) for i in item[0].split(" ")]
			if pos_edge[0] not in nodes_neibor:
				nodes_neibor[pos_edge[0]] = [pos_edge[1]]
			else:
				nodes_neibor[pos_edge[0]].append(pos_edge[1])
			if pos_edge[1] not in nodes_neibor:
				nodes_neibor[pos_edge[1]] = [pos_edge[0]]
			else:
				nodes_neibor[pos_edge[1]].append(pos_edge[0])

		return nodes_neibor

	# 产生负例link
	def gen_edge_data(self,edgefilepath,node_neibor):

		edgefile = open(edgefilepath,"r")
		pos_edges = []
		for line in edgefile.readlines():
			item = line.strip().split("#")
			pos_edge = [int(i) for i in item[0].split(" ")]
			pos_edges.append(pos_edge)

		neg_edges = []
		for pos_edge in pos_edges:
			neg = np.random.choice(self.nodes_num)
			while neg in node_neibor[pos_edge[0]]:
				neg = np.random.choice(self.nodes_num)
			neg_edges.append([pos_edge[0],neg])

		edges = pos_edges + neg_edges

		# edges_predict = metrics.pairwise.cosine_similarity(np.array(self.nodes_emb)[np.array(edges)[:,0],:],\
		# 	np.array(self.nodes_emb)[np.array(edges)[:,1],:]).diagonal()
		
		data_size = len(edges)
		batch_size = 1000
		edges_predict = []
		start_index = 0
		end_index = min(start_index + batch_size,data_size)
		indices = np.arange(data_size)
		while start_index < data_size:
			index = indices[start_index:end_index]
			h_batch = np.array(self.nodes_emb)[np.array(edges)[index][:,0],:]
			t_batch = np.array(self.nodes_emb)[np.array(edges)[index][:,1],:]
			edges_predict += metrics.pairwise.cosine_similarity(h_batch,t_batch).diagonal().tolist()

			start_index = end_index
			end_index = min(start_index + batch_size,data_size)  

		edges_label = [1]*len(pos_edges) + [0]*len(neg_edges)

		return edges_predict,edges_label
			
	# 节点分类函数
	def node_classfication(self):

		acc_micro_list = []
		acc_macro_list = []
		self.nodes_emb = np.array(self.nodes_emb)[range(self.nodes_num)]
		for test_size in [0.85,0.75,0.65,0.55,0.45,0.35,0.25]:
			x_train,x_test,y_train,y_test = train_test_split(self.nodes_emb,\
				self.nodes_label, random_state=1, test_size=test_size)
			clf = svm.SVC(C=10,kernel='rbf')
			clf.fit(x_train,y_train)
			y_test_hat = clf.predict(x_test)
			acc_micro_list.append(str(np.round(metrics.f1_score(y_test,y_test_hat,average='micro')*10000)/100))
			acc_macro_list.append(str(np.round(metrics.f1_score(y_test,y_test_hat,average='macro')*10000)/100))
		print("node classfication...")
		self.resultfile.write("\tnode classfication:\n")
		self.resultfile.write("\t\tF1-score(micro): %s\n"%(" ".join(acc_micro_list)))
		self.resultfile.write("\t\tF1-score(macro): %s\n"%(" ".join(acc_macro_list)))

	# 节点聚类函数
	def node_clustering(self):
		
		cate_num = len(set(self.nodes_label))
		clf = KMeans(n_clusters=cate_num,init="k-means++")
		kmeans = clf.fit(self.nodes_emb)
		cluster_groups = kmeans.labels_
		acc = metrics.adjusted_rand_score(self.nodes_label,cluster_groups)
		nmi = metrics.normalized_mutual_info_score(self.nodes_label,cluster_groups)
		print("node clustering...")
		self.resultfile.write("\tnode clustering:\n")
		self.resultfile.write("\t\tacc: %.4f, nmi: %.4f\n"%(acc,nmi))

	# 网络可视化函数
	def network_visualization(self):

		print("network visualization...")
		nodes_label = np.array(self.nodes_label)
		emb_tsne = TSNE().fit_transform(self.nodes_emb)
		plt.scatter(emb_tsne[:,0],emb_tsne[:,1],10*nodes_label,10*nodes_label)
		plt.show()

	# 链路预测函数
	def link_prediction(self,edgefilepath):

		node_neibor = self.read_edgefile(edgefilepath)
		y_predict,y_true = self.gen_edge_data(edgefilepath+"_test"+str(np.round((1.0-self.train_size)*100)),node_neibor)

		auc = metrics.roc_auc_score(y_true,y_predict)
		if auc < 0.5:
			auc = 1.0 - auc

		print("link prediction...")
		self.resultfile.write("%.4f "%auc)
		if(self.train_size == 0.8):
			self.resultfile.write("\n\n")
		# print(auc)

	# 节点推荐函数
	def node_recommendation(self,edgefilepath):

		nodes_neibor = self.read_edgefile(edgefilepath)
		if(self.nodes_num == self.nodes_num_total):
			consine_mat = metrics.pairwise.cosine_similarity(np.array(self.nodes_emb),np.array(self.nodes_emb))
			nodes_range = range(self.nodes_num)
		else:
			consine_mat = metrics.pairwise.cosine_similarity(np.array(self.nodes_emb)[self.nodes_num:],\
				np.array(self.nodes_emb)[range(self.nodes_num)])
			nodes_range = range(self.nodes_num_total-self.nodes_num)
		recall = []
		precision = []
		for hitcount in [10,20,50]:
			hit = np.argsort(consine_mat,axis=1)[:,-(1+hitcount):-1]
			hit_num = self.nodes_num * hitcount
			neibor_num = 0
			target_num = 0
			for node in nodes_range:
				if(self.nodes_num == self.nodes_num_total):
					neibor = nodes_neibor[node]
				else:
					neibor = nodes_neibor[node+self.nodes_num]
				target = set(hit[node]).intersection(set(neibor))
				neibor_num += len(neibor)
				target_num += len(target)
			recall.append(str(np.round(float(target_num)/float(neibor_num)*10000)/100))
			precision.append(str(np.round(float(target_num)/float(hit_num)*10000)/100))
		print("node recommendation...")
		self.resultfile.write("\tnode recommendation:\n")
		self.resultfile.write("\t\trecall: %s\n"%(" ".join(recall)))
		self.resultfile.write("\t\tprecision: %s\n"%(" ".join(precision)))

	# 相似节点查找函数
	def similarity_search(self):

		consine_mat = metrics.pairwise.cosine_similarity(np.array(self.nodes_emb)[range(self.nodes_num)],\
			np.array(self.nodes_emb)[range(self.nodes_num)])
		hitcount = 20
		hit = np.argsort(consine_mat,axis=1)[:,-(1+hitcount):-1]

		recall = []
		precision = []
		print("similarity search...")
		for hitcount in [10,20,50]:
			hit = np.argsort(consine_mat,axis=1)[:,-(1+hitcount):-1]
			hit_num = self.nodes_num * hitcount
			target_num = 0
			for node in range(self.nodes_num):
				sim_label = np.array(self.nodes_label)[hit[node]]
				target_num += np.sum(sim_label==self.nodes_label[node])
			precision.append(str(np.round(float(target_num)/float(hit_num)*10000)/100))
		self.resultfile.write("\tsimilarity search:\n")
		self.resultfile.write("\t\tprecision: %s\n"%(" ".join(precision)))
