#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[155]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[7]:


movies.head()


# In[8]:


credits.head()


# In[10]:


credits.head(1)['cast'].values


# In[156]:


movies.merge(credits,on='title')


# In[157]:


movies.merge(credits,on='title').shape


# In[20]:


movies.shape


# In[21]:


credits.shape


# In[158]:


movies=movies.merge(credits,on='title')


# In[24]:


movies.info()


# In[159]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[160]:


movies.head()


# In[161]:


movies.isnull().sum()


# In[162]:


movies.dropna(inplace=True)


# In[163]:


movies.isnull().sum()


# In[164]:


movies.duplicated().sum()


# In[165]:


movies.iloc[1].genres


# In[46]:


import ast


# In[47]:


def covert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        #string indices must be integers
        L.append(i['name'])
    return L    


# In[166]:


movies['genres']=movies['genres'].apply(covert)


# In[56]:


movies['keywords']


# In[167]:


movies['keywords']=movies['keywords'].apply(covert)


# In[57]:


movies.head()


# In[60]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break    
    return L


# In[168]:


movies['cast']=movies['cast'].apply(convert3)


# In[169]:


def crewconverter(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


# In[170]:


movies['crew']=movies['crew'].apply(crewconverter)


# In[84]:


movies['overview'][0]


# In[171]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[172]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keyword']=movies['overview'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[88]:


movies.head()


# In[173]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['cast']


# In[135]:


movies.head()


# In[175]:


#movies.drop("tag", axis='columns')


# In[176]:


new_df=movies[['movie_id','title','tags']]
new_df


# In[177]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))#converting tags from list to string


# In[178]:


new_df['tags'][0]


# In[179]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[141]:


new_df.head()


# In[142]:


#inorder to do recommentation, we convert tags and perform vectorisation


# In[180]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[181]:


vector=cv.fit_transform(new_df['tags']).toarray()
#made a matrix


# In[182]:


cv.get_feature_names()


# In[183]:


vector


# In[184]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[185]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[148]:


new_df['tags'][0]


# In[149]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction culture clash future space war space colony society space travel futuristic romance space alien tribe alien planet cgi marine soldier battle love affair anti war power relations mind and soul 3d samworthington zoesaldana sigourneyweaver samworthington zoesaldana sigourneyweaver')


# In[186]:


new_df['tags']=new_df['tags'].apply(stem)


# In[187]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[188]:


vector=cv.fit_transform(new_df['tags']).toarray()


# In[190]:


vector


# In[191]:


cv.get_feature_names()


# In[192]:


#we find ditance btw each vector with all the other vectors using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity


# In[193]:


cosine_similarity(vector)


# In[194]:


cosine_similarity(vector).shape


# In[195]:


similarity=cosine_similarity(vector)


# In[225]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distance=similarity[movie_index]
    movie_list=sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movie_list:
        print (new_df.iloc[i[0]].title)


# In[226]:


recommend('Batman Begins')


# In[228]:


recommend('Avatar')

