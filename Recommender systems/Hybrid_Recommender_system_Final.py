#!/usr/bin/env python
# coding: utf-8

# In[12]:


import snscrape.modules.twitter as sntwitter
import pandas as pd
import snscrape.modules.twitter as snt
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[2]:


collected_tweets = []


# In[73]:


mov_data=pd.read_csv("C:/Users/ACER/Documents/Social Media Project/Movielens2.csv")


# In[34]:


len(mov_data)


# In[5]:


tags = mov_data.movie_title


# # Scraping Tweets

# In[8]:


#Scraping
for tag in tags:
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(tag+' since:2015-01-01 until:2022-07-30 lang:en').get_items()):
        if i>50:
            break
        else:
            collected_tweets.append([tweet.user.username,tag,tweet.content])
    tweets_data = pd.DataFrame(collected_tweets,columns=["User","Moviename","Content"])


# In[9]:


tweets_data #Tweets collected


# In[11]:


tweets_data.to_csv('Collected1Tweets.csv')


# In[13]:


stopwordsenglish = stopwords.words('English')
def cleanData(text):
    text = re.sub(r'@[A-Za-z0-9]+', ' ', text)  # Removes @ with text
    text = re.sub(r'https?:\/\/\S+', ' ', text)  # Removes the Hyperlink
    text = re.sub(r'&[A-Za-z0-9]+', ' ', text)  # Removes & with text 
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\W', ' ', text)  # Removes all punctuations and symbols - also removes #
    text = re.sub(r'[0-9]', '', text)  # Removes all numbers
    text = re.sub(r'_', ' ', text)
    text = re.sub(r' +', ' ', text)  # Removes more than one or more space from the tweets
    text = text.lower()              #lowercase
    text_tokenize = word_tokenize(text)
    text_wo_stopwords = []
    for word in text_tokenize:
        if word not in stopwordsenglish:
            text_wo_stopwords.append(word)
    return (' '.join(text_wo_stopwords))


# In[16]:


tweet_clean=[]
for tweet in tweets_data.Content:
    tweet_clean.append(cleanData(tweet))


# In[17]:


tweet_clean #Cleaned Tweets 


# In[18]:


tweets_data['clean_content'] = tweet_clean


# In[19]:


tweets_data


# In[20]:


tweets_data.to_csv('CollectedcleanTweets.csv')


# # Calculating Sentiment Score

# In[ ]:


tweets_data.groupby(tweets_data['Moviename'])tweets_data.clean_content


# In[22]:


analyzer = SentimentIntensityAnalyzer()
vs = [analyzer.polarity_scores(tweetu)['compound'] for tweetu in tweets_data.clean_content]


# In[23]:


len(vs)


# In[24]:


tweets_data['sentiment_score'] = vs


# In[72]:


mov_data


# In[57]:


sn8=[]


# In[59]:


sn8=tweets_data.groupby(['Moviename'])['sentiment_score'].mean()


# In[76]:


mov_data = mov_data.sort_values(by='movie_title')


# In[77]:


mov_data


# In[63]:


sn9=pd.DataFrame({'movie_id':sn8.index, 'sentiment_score':sn8.values})


# In[363]:


sn9


# In[78]:


mov_data['sentiment_scor'] = mov_data['movie_title'].map(sn9.set_index('movie_id')['sentiment_score'].drop_duplicates())


# In[82]:


mov_data.to_csv('moovie_lenswithsentiment.csv')


# In[417]:


df=pd.read_csv("C:/Users/ACER/Documents/Social Media Project/moovie_lenswithsentiment.csv")


# In[418]:


df


# In[370]:


ratings_df = pd.read_csv("C:/Users/ACER/Documents/Social Media Project/rating_data.data",sep='\t',names=["user_id","item_id","rating","timestamp"])


# In[371]:


ratings_df


# In[372]:


sn8=ratings_df.groupby(['item_id'])['rating'].mean()


# In[373]:


sn8


# In[374]:


sn9=pd.DataFrame({'movie_id':sn8.index, 'rating':sn8.values})


# In[375]:


sn9


# In[376]:


sn9['sentiment_score'] = sn9['movie_id'].map(mov_data.set_index('movie_id')['sentiment_scor'].drop_duplicates())


# In[377]:


ratings_df['sentiment_score'] = ratings_df['item_id'].map(sn9.set_index('movie_id')['sentiment_score'])


# In[378]:


ratings_df


# In[109]:


#ratings_df.to_csv("ratings_df_updated.csv")


# In[449]:


ratings_df=pd.read_csv("C:/Users/ACER/Documents/Social Media Project/ratings_df_updated.csv")


# In[450]:


ratings_df


# In[119]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[380]:


train_data,vaild_data_df = train_test_split(df,test_size=0.2)


# In[381]:


df


# # Building Dataset for hybrid Recommender System

# In[451]:


df_1 = df[['movie_id','movie_title',' unknown',' Action',' Adventure',' Animation',"Children's",' Comedy',' Crime',' Documentary',
          ' Drama',' Fantasy','Film-Noir',' Horror',' Musical',' Mystery',' Romance',' Sci-Fi','Thriller',' War ',' Western ']]
     


# In[452]:


ratings_df


# In[453]:


ratings_df.rename(columns={'item_id':'movie_id'},inplace=True)


# In[465]:


ratings_df


# In[455]:


result_data = pd.merge(ratings_df,df_1, how="inner", on=["movie_id"])
result_data


# In[456]:


lc = LabelEncoder()
result_data['movie_title'] = lc.fit_transform(result_data['movie_title'])
result_data


# In[425]:


#result_data=result_data.drop('timestamp', axis=1, inplace=True)


# In[426]:


result_data


# In[409]:


print(result_data)


# In[457]:


train_data,vaild_data_df = train_test_split(result_data,test_size=0.2)


# In[458]:


num_users_hybrid = len(train_data.user_id)
num_movie_items_hybrid = len(train_data.movie_id)
print(num_users_hybrid, num_movie_items_hybrid) 


# # Hybrid recommender System

# In[459]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[460]:


class NeuralNet(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100, n_hidden=10):
        super(NeuralNet, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.layer1 = nn.Linear(emb_size*2 + n_hidden+n_hidden, n_hidden)
        self.layer2 = nn.Linear(n_hidden, 1)
        self.drop1 = nn.Dropout(0.1)
        
    def forward(self, u, v):
        user = self.user_emb(u)
        movie = self.item_emb(v[:,1])
        movie_features = v[:,4:]
        x = F.relu(torch.cat([user, movie,movie_features], dim=1))
        x = self.drop1(x)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


# In[461]:


def train_epocs_hybrid(model, epochs=10, lr=0.01, wd=0.0, unsqueeze=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for i in range(epochs):
        users = torch.LongTensor(train_data.user_id.values) # .cuda()
        items = torch.LongTensor(train_data.values) #.cuda()
        ratings = torch.FloatTensor(train_data.rating.values) #.cuda()
        if unsqueeze:
            ratings = ratings.unsqueeze(1)
        y_hat = model(users, items)
        loss = F.mse_loss(y_hat, ratings)
        train_mse_hybrid.append((i, loss.item()))
        if (i+1) % 10 == 0:
          print("Iteration: %d ; error = %.4f" % (i+1, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    test_loss_hybrid(model, unsqueeze)


# In[462]:


def test_loss_hybrid(model, unsqueeze=False):
    model.eval()
    users = torch.LongTensor(valid_data_df.user_id.values) #.cuda()
    items = torch.LongTensor(valid_data_df.values) #.cuda()
    ratings = torch.FloatTensor(valid_data_df.rating.values) #.cuda()
    if unsqueeze:
        ratings = ratings.unsqueeze(1)
    y_hat = model(users, items)
    loss = F.mse_loss(y_hat, ratings)
    print("test loss %.3f " % loss.item())


# In[477]:


model_hybrid_23 = NeuralNet(num_users_hybrid, num_movie_items_hybrid, emb_size=20)
model_hybrid_23


# In[478]:


train_mse_hybrid = []
train_epocs_hybrid(model_hybrid_23, epochs=20, lr=0.01, wd=1e-6, unsqueeze=True)


# In[479]:


import matplotlib.pyplot as plt


# In[480]:


x = [x for x, y in train_mse_hybrid]
y = [y for x, y in train_mse_hybrid]
plt.figure(figsize=((4,4)))
plt.plot(x, y)
plt.xticks(x, x)
plt.title("k = 20")
plt.xlabel("epochs")
plt.ylabel("RMSE")
plt.grid(axis="y")


# # Hybrid Model with Latent feature K=20

# In[481]:


npRecommend_20 = result_data[result_data.user_id == 5].to_numpy()


# # Top 20 recommendation for user 5

# In[482]:


npRecommend_val_20 = model_hybrid_23(torch.LongTensor(npRecommend_20[:,0]),torch.LongTensor(npRecommend_20)).detach().numpy().reshape(-1)
npRecommend_val_20


# In[484]:


import numpy as np


# In[485]:


indecies_20 =  np.argsort(npRecommend_val_20)[-20:][::-1]
indecies_20


# In[488]:


recommded_itemid_20_hybrid = result_data['movie_id'].unique()[indecies_20]  # taking top 10
print(recommded_itemid_20_hybrid)


# In[491]:


#!pip install plotly


# In[492]:


import plotly.express as px


# In[495]:


recommendedMovies_20_hybrid = result_data[result_data.movie_id.isin(recommded_itemid_20_hybrid)]
top20k_hybrid = pd.DataFrame(recommendedMovies_20_hybrid['movie_title'])
top20k_hybrid =top20k_hybrid.reset_index(drop=True)
top20k_hybrid = top20k_hybrid.reset_index()
top20k_hybrid = top20k_hybrid.rename(columns={'index':'top 20'})
px.line(top20k_hybrid,x='top 20',y='movie_title',width=600)


# In[496]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# # Evaluation Metrics

# In[497]:


y_pred_user5 = npRecommend_val_20 #model with k=20
y_true_user5 = npRecommend_20[:,2]


# # RMSE AND MAE

# In[498]:


mean_squared_error(y_true_user5,y_pred_user5)


# In[499]:


mean_absolute_error(y_pred_user5,y_true_user5)


# In[ ]:




