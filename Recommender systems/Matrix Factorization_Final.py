#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from collections import Counter
from sklearn.model_selection import train_test_split
from scipy import sparse


# Loading Dataset

# In[6]:


ratings_df = pd.read_csv("C:/Users/ACER/Documents/Social Media Project/rating_data.data",sep='\t',names=["user_id","item_id","rating","timestamp"])


# In[7]:


ratings_df


# In[8]:


ratings_df = ratings_df.reset_index()[['user_id','item_id','rating']]
print(ratings_df.shape)


# Distribution of ratings

# In[9]:


Counter(ratings_df.rating)


# Number of ratings per user

# In[11]:


Counter(ratings_df.groupby(['user_id']).count()['item_id'])


# Train-Test split

# In[14]:


train_df, valid_df = train_test_split(ratings_df, test_size=0.2)

#resetting indices to avoid indexing errors in the future
train_df = train_df.reset_index()[['user_id', 'item_id', 'rating']]
valid_df = valid_df.reset_index()[['user_id', 'item_id', 'rating']]


# Training

# In[15]:


def encode_column(column):
    """ Encodes a pandas column with continous IDs"""
    keys = column.unique()
    key_to_id = {key:idx for idx,key in enumerate(keys)}
    return key_to_id, np.array([key_to_id[x] for x in column]), len(keys)


# In[16]:


def encode_df(ratings_df):
    """Encodes rating data with continuous user and anime ids"""
    
    item_ids, ratings_df['item_id'], num_items = encode_column(ratings_df['item_id'])
    user_ids, ratings_df['user_id'], num_users = encode_column(ratings_df['user_id'])
    return ratings_df, num_users, num_items, user_ids, item_ids


# In[18]:


ratings_df, num_users, num_items, user_ids, item_ids = encode_df(train_df)
print("Number of users :", num_users)
print("Number of Movies :", num_items)
ratings_df.head()


# Initializing user and item embeddings

# In[19]:


def create_embeddings(n, K): #Function to add embeddings
    """
    Creates a random numpy matrix of shape n, K with uniform values in (0, 11/K)
    n: number of items/users
    K: number of factors in the embedding 
    """
    return 11*np.random.random((n, K)) / K


# In[20]:


def create_sparse_matrix(df, rows, cols, column_name="rating"): #Sparse matrix
    """ Returns a sparse utility matrix""" 
    return sparse.csc_matrix((df[column_name].values,(df['user_id'].values, df['item_id'].values)),shape=(rows, cols))


# Creating Sparse Matrix

# In[21]:


ratings_df, num_users, num_items, user_ids, item_ids = encode_df(train_df)
Y = create_sparse_matrix(ratings_df, num_users, num_items)


# In[22]:


# to view matrix
Y.todense()


# Making predictions

# In[23]:


def predict(df, emb_user, emb_item):
    """ This function computes df["prediction"] without doing (U*V^T).
    
    Computes df["prediction"] by using elementwise multiplication of the corresponding embeddings and then 
    sum to get the prediction u_i*v_j. This avoids creating the dense matrix U*V^T.
    """
    df['prediction'] = np.sum(np.multiply(emb_item[df['item_id']],emb_user[df['user_id']]), axis=1)
    return df


# In[24]:


lmbda = 0.0002


# In[25]:


def cost(df, emb_user, emb_item):
    """ Computes mean square error"""
    Y = create_sparse_matrix(df, emb_user.shape[0], emb_item.shape[0])
    predicted = create_sparse_matrix(predict(df, emb_user, emb_item), emb_user.shape[0], emb_item.shape[0], 'prediction')
    return np.sum((Y-predicted).power(2))/df.shape[0] 


# Gradient Descent

# In[26]:


def gradient(df, emb_user, emb_item):
    """ Computes the gradient for user and anime embeddings"""
    Y = create_sparse_matrix(df, emb_user.shape[0], emb_item.shape[0])
    predicted = create_sparse_matrix(predict(df, emb_user, emb_item), emb_user.shape[0], emb_item.shape[0], 'prediction')
    delta =(Y-predicted)
    grad_user = (-2/df.shape[0])*(delta*emb_item) + 2*lmbda*emb_user
    grad_item = (-2/df.shape[0])*(delta.T*emb_user) + 2*lmbda*emb_item
    return grad_user, grad_item


# In[27]:


def gradient_descent(df, emb_user, emb_item, iterations=2000, learning_rate=0.01, df_val=None):
    """ 
    Computes gradient descent with momentum (0.9) for given number of iterations.
    emb_user: the trained user embedding
    emb_anime: the trained anime embedding
    """
    Y = create_sparse_matrix(df, emb_user.shape[0], emb_item.shape[0])
    beta = 0.9
    grad_user, grad_item = gradient(df, emb_user, emb_item)
    v_user = grad_user
    v_item = grad_item
    for i in range(iterations):
        grad_user, grad_item = gradient(df, emb_user, emb_item)
        v_user = beta*v_user + (1-beta)*grad_user
        v_item = beta*v_item + (1-beta)*grad_item
        emb_user = emb_user - learning_rate*v_user
        emb_item = emb_item - learning_rate*v_item
        if(not (i+1)%50):
            print("\niteration", i+1, ":")
            print("train mse:",  cost(df, emb_user, emb_item))
            if df_val is not None:
                print("validation mse:",  cost(df_val, emb_user, emb_item))
    return emb_user, emb_item


# In[28]:


emb_user = create_embeddings(num_users, 3)
emb_item = create_embeddings(num_items, 3)
emb_user, emb_item = gradient_descent(ratings_df, emb_user, emb_item, iterations=800, learning_rate=1)


# Making predictions on new data

# In[30]:


def encode_new_data(valid_df, user_ids, item_ids):
    """ Encodes valid_df with the same encoding as train_df.
    """
    df_val_chosen = valid_df['item_id'].isin(item_ids.keys()) & valid_df['user_id'].isin(user_ids.keys())
    valid_df = valid_df[df_val_chosen]
    valid_df['item_id'] =  np.array([item_ids[x] for x in valid_df['item_id']])
    valid_df['user_id'] = np.array([user_ids[x] for x in valid_df['user_id']])
    return valid_df


# In[31]:


print("before encoding:", valid_df.shape)
valid_df = encode_new_data(valid_df, user_ids, item_ids)
print("after encoding:", valid_df.shape)


# Mean Square Error

# In[32]:


train_mse = cost(train_df, emb_user, emb_item)
val_mse = cost(valid_df, emb_user, emb_item)
print(train_mse, val_mse)

