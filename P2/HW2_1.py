#!/usr/bin/env python
# coding: utf-8

# In[33]:


import h5py 
import numpy as np
#import math

f = h5py.File('mnist_testdata.hdf5','r') 
#f=h5py.File('mnist_network_params.hdf5','r')
list(f.keys())
#list(f1.keys())


# In[34]:


xtrain=f['xdata']
#ytrain=f['ydata']


# In[35]:


f1 = h5py.File('mnist_network_params.hdf5','r') 
w1=f1['W1']
w2=f1['W2']
w3=f1['W3']
b1=f1['b1']
b2=f1['b2']
b3=f1['b3']


# In[36]:


print(w1.shape,w2.shape,w3.shape,b1.shape,b2.shape,b3.shape)
w1=np.array(w1)
w2=np.array(w2)
w3=np.array(w3)
b1=np.array(b1)
b2=np.array(b2)
b3=np.array(b3)
xtrain=np.array(xtrain)
#ytrain=np.array(ytrain)


# In[37]:


def relu(x):
    return np.maximum(x, 0)


# In[38]:


def Softmax(x):
    exp_scores = np.exp(x)
    out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return out


# In[39]:


def feed_forward(x):
    z1 = x@w1.transpose() + b1
    a1 = relu(z1)
    z2 = a1@w2.transpose() + b2
    a2 = relu(z2)
    z3 = a2@w3.transpose() + b3
    a3 = relu(z3)
    out = Softmax(a3)
    return a1,a2,a3,out


# In[43]:


pred=feed_forward(xtrain)
pred_result=pred[3]
print(pred[3][0:3,:])


# In[41]:


num=0
n_iter=xtrain.shape[0]
pred_result=np.zeros((n_iter,10),float)
pred_label=np.zeros((n_iter,1),float)
for i in range (0,n_iter):
    pred_result[i,:]=pred[3][i,:]
    label=np.where(pred_result[i,:]==np.max(pred_result[i,:]))   # find label for each predict data
    pred_label[i,0]=label[0][0]
#    train_result=ytrain[i,:]
#    index_1=train_result.nonzero()[0]
#    num_1=pred_result[index_1[0]]
#    if np.equal(pred_result.max(0),num_1):
#        num=num+1    
#print(num)
#print(num_false)


# In[46]:


output = []
for i in range(0,n_iter):
    output.append({
      "activations": pred_result[i,:].tolist(),
      "index": i,
      "classification": int(pred_result[i,:].argmax())
    })

import json
print("AUTOGRADE: %s" % (json.dumps(output)))


# In[ ]:




