#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[12]:


movies_data = pd.read_csv(r'C:\Users\Srihari\Downloads\movies.csv')


# In[13]:


movies_data.head()


# In[14]:


movies_data.shape


# In[15]:


selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)


# In[16]:


for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')


# In[17]:


combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']


# In[18]:


print(combined_features)


# In[19]:


vectorizer = TfidfVectorizer()


# In[20]:


feature_vectors = vectorizer.fit_transform(combined_features)


# In[21]:


print(feature_vectors)


# In[22]:


similarity = cosine_similarity(feature_vectors)


# In[23]:


print(similarity)


# In[24]:


print(similarity.shape)


# In[25]:


# getting the movie name from the user

movie_name = input(' Enter your favourite movie name : ')


# In[26]:


list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)


# In[27]:


# finding the close match for the movie name given by the user

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)


# In[28]:


close_match = find_close_match[0]
print(close_match)


# In[29]:


index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)


# In[30]:


similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


# In[31]:


len(similarity_score)


# In[32]:


# sorting the movies based on their similarity score

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(sorted_similar_movies)


# In[33]:


# print the name of similar movies based on the index

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


# In[34]:


movie_name = input(' Enter your favourite movie name : ')

list_of_all_titles = movies_data['title'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


# In[ ]:




