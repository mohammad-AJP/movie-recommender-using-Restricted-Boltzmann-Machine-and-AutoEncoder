#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#%% importing the dataset
movies = pd.read_csv("ml-1m/movies.dat" , sep = "::" , header = None , engine ="python",
                     encoding = "latin-1")
users = pd.read_csv("ml-1m/users.dat" , sep = "::" , header = None , engine ="python",
                     encoding = "latin-1")
ratings = pd.read_csv("ml-1m/ratings.dat" , sep = "::" , header = None , engine ="python",
                     encoding = "latin-1")

#%% preparing test set and training set
training_set = pd.read_csv("ml-100k/u1.base" , delimiter = "\t")
training_set = np.array(training_set , dtype = "int")
test_set = pd.read_csv("ml-100k/u1.test" , delimiter = "\t")
test_set = np.array(test_set , dtype = "int")
#%%  getting the number of users and movies
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))
#%% converting data into arrays in which users are lines and movies are columns
def convert(data):
    new_data = []
    for id_user in range(1 , nb_users+1):
        movies = data[:,1][data[: , 0] == id_user]
        ratings = data[: , 2][data[:,0] == id_user]
        list1 = np.zeros(nb_movies)
        list1[movies-1] = ratings
        new_data.append(list(list1))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

#%% converting data into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#%% converting the ratings into binary ratings 1 (liked) and 0 (disliked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

#%% creating the architecture of the RBM
class RBM():
    def __init__(self,nv,nh):
        self.W = torch.randn(nh , nv)
        self.a = torch.randn(1 , nh)#biases of hidden nodes
        self.b = torch.randn(1 , nv)#bias of visible nodes
    def sample_h(self, x):
        wx = torch.mm(x , self.W.t())#transposing bacause W is weights for visible nodes given hidden nodes
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v , torch.bernoulli(p_h_given_v)#it is for sampling hidden neurons
    def sample_v(self, y):
        wy = torch.mm(y , self.W) # we do not need transpose because the direction is fron visible to hidden
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h , torch.bernoulli(p_v_given_h)#it is for sampling visible neurons
    def train(self, v0, vk, ph0, phk):
        #self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)# t() is to match mat mathematically
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)# to keep the dimensions as we want
        self.a += torch.sum((ph0 - phk), 0)# to keep the dimensions as we wanted

nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv , nh)

#%% training the rbm
nb_epoch = 20
for epoch in range(1 , nb_epoch + 1):
    train_loss = 0
    s = 0. # counter that increases after each epoch and we use it to normalize train_loss(0. is because it is float)
    for id_user in range (0 , nb_users - batch_size, batch_size):# we are applying batch sizes
        vk = training_set[id_user : id_user+batch_size]# output is initially those part of training_set that we are fetching
        v0 = training_set[id_user : id_user+batch_size]
        ph0, _ = rbm.sample_h(v0)# _ is to tell that we just need the first argument of sample_h func
        for k in range(10): # steps of gibbs sampling
            _, hk = rbm.sample_h(vk)# vk because v0 is fixed (it is our target so shouldnt change)
            _, vk = rbm.sample_v(hk)# visible nodes getting updated
            vk[v0<0] = v0[v0<0] # to prevent the nodes that are not rated by users from upadating
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0 >=0] - vk[v0 >=0]))
        s += 1.
    print('epoch: '+str(epoch)+" loss: "+str(train_loss/s))
    
#%% testing the rbm
test_loss = 0
s = 0. 
for id_user in range (nb_users):# we are applying batch sizes
    v = training_set[id_user : id_user + 1]# training_set is used to activate hidden nodes and predict of unrated nodes
    vt = test_set[id_user : id_user + 1]
    if len(vt[vt>=0]) > 0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt >=0] - v[vt >=0]))
        s += 1.
print("test_loss: "+str(test_loss/s))
            
            
            
        
        


























        