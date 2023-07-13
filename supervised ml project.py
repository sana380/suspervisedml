#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install nltk')


# In[1]:


import nltk
nltk.download()


# In[10]:


import nltk
nltk.sent_tokenize("Sample sentence for tokenization.")


# In[11]:


get_ipython().system('pip install scikit-learn')


# In[ ]:


import sklearn
print(sklearn.__version__)


# In[2]:


import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# In[3]:


# Step 1: Preprocess the data and load the labeled dataset
positive_reviews = [ "I loved the movie, it was fantastic!",
    "The restaurant had amazing food and great service.",
    "The book was captivating and well-written."] # List of positive reviews
negative_reviews = [  "The movie was terrible, I wouldn't recommend it.",
    "I had a bad experience at the restaurant, the food was cold and the service was rude.",
    "The book was disappointing and poorly written."
]  # List of negative reviews


# In[4]:


# Step 2: Create labels for the reviews (1 for positive, 0 for negative)
reviews = positive_reviews + negative_reviews
labels = np.concatenate([np.ones(len(positive_reviews)), np.zeros(len(negative_reviews))])


# In[5]:


# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=42)


# In[6]:


# Step 4: Convert the reviews into numerical feature vectors using the bag-of-words model
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# In[7]:


# Step 5: Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)


# In[8]:


# Step 6: Make predictions on the testing set
y_pred = classifier.predict(X_test)


# In[9]:


# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:




