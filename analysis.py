# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: myenv
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os
import spacy
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from scipy.sparse import hstack
from scipy.sparse import csr_matrix


# %%
nlp = spacy.load('en_core_web_sm')

# %%
df = pd.read_csv("reduced_spotify_data.csv")
df.head()

# %%
df = df[['text', 'Genre', 'Tempo', 'emotion']].dropna()

# %% [markdown]
# Vectorizing the lyrics

# %%
df_sample['vec'] = df_sample['text'].apply(lambda doc: nlp(doc).vector)

# %% [markdown]
# Feature matrix with vectorized values

# %%
X_lyrics = np.vstack(df_sample['vec'].values)

# %% [markdown]
# Encode genres

# %%
ohe = OneHotEncoder()
genre = ohe.fit_transform(df_sample[['Genre']])

# %%
tempo = csr_matrix(df_sample[['Tempo']].values)

# %%
combined = hstack([csr_matrix(X_lyrics), X_genre, X_tempo])


# %% [markdown]
# split to train

# %%
y = df_sample['emotion']
X_train, X_test, y_train, y_test = train_test_split(combined, y, test_size=0.2, random_state=42)

# %% [markdown]
# try logistic regression

# %%
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# %% [markdown]
# Finding null values to clean data

# %%
df.isnull().values.any()
df.isnull().head()
nan_count = np.sum(df.isnull(), axis = 0)
print(nan_count)

# %%
sns.countplot(x=y_pred)
plt.title("Predicted Emotion Distribution")
plt.xticks(rotation=45)

# %%
df_sample.shape
