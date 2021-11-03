#%% AuotEncoders
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#%% importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

#%% preparing test set and training set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

#%%  getting the number of users and movies
nb_users = int(max(max(training_set[:, 0], ), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1], ), max(test_set[:, 1])))

#%% converting data into arrays in which users are lines and movies are columns
def convert(data):
  new_data = []
  for id_users in range(1, nb_users + 1):
    id_movies = data[:, 1] [data[:, 0] == id_users]
    id_ratings = data[:, 2] [data[:, 0] == id_users]
    ratings = np.zeros(nb_movies)
    ratings[id_movies - 1] = id_ratings
    new_data.append(list(ratings))
  return new_data
training_set = convert(training_set)
test_set = convert(test_set)

#%% converting data into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#%% creating the architecture of AE
class SAE(nn.Module): # we are inheritting from nn module of pytorch
    def __init__(self, ):
        #the first layer which is represented by object with self. :
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20) # no more encoding but decoding
        self.fc4 = nn.Linear(20, nb_movies)
        # we now define an activation function
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x)) # encoding
        x = self.activation(self.fc2(x)) # encoding
        x = self.activation(self.fc3(x)) # encoding
        x = self.fc4(x) # last layer and we do not need any activation function to be applied
        return x
sae = SAE() # an autoencoder objects from SAE class
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

#%% training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
  train_loss = 0 # it will be updated in epochs
  s = 0. # defined to calculate just for users that have rated at least one movie(it is used is RMSprop so it should be float)
  for id_user in range(nb_users): # to loop over all users
    input_1 = Variable(training_set[id_user]).unsqueeze(0) # to shape the training_set as what pytorch accepts
    target = input_1.clone() # make a copy of input_1
    if torch.sum(target.data > 0) > 0: # to save memory, we just take actions on users that have rated at least one movie
      output = sae(input_1) #inside sae, we have def. forward that calcuates estimated output so we use it here
      target.require_grad = False #do not calculate GD for targe and save calculation
      output[target == 0] = 0 # do not calculate outputs which their real values are zero(not rated)
      loss = criterion(output, target)
      mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) #1e-10 to prevent the nominator to be zero
      loss.backward() #defines the direction of the loss
      train_loss += np.sqrt(loss.data*mean_corrector)
      s += 1.
      optimizer.step() #calculate the amount(intensity) of update
  print('epoch: '+str(epoch)+'loss: '+ str(train_loss/s))

#%% Testing the AE
test_loss = 0
s = 0.
for id_user in range(nb_users):
  input = Variable(training_set[id_user]).unsqueeze(0)
  target = Variable(test_set[id_user]).unsqueeze(0)
  if torch.sum(target.data > 0) > 0:
    output = sae(input)
    target.require_grad = False
    output[target == 0] = 0
    loss = criterion(output, target)
    mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
    test_loss += np.sqrt(loss.data*mean_corrector)
    s += 1.
print('test loss: '+str(test_loss/s))

