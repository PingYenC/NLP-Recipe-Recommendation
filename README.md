# NLP-Recipe-Reccomendation
Use TFIDF + CountVectorizer to compare users' query and recipe, come up with recommendations
The data and model are from and stored in my google drive

# importing inportant libraries
import os

import math

import datetime

from tqdm import tqdm

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import re

from sklearn.decomposition import TruncatedSVD

from gensim.models import Word2Vec

import gensim.utils

import gensim.downloader as API

from gensim.utils import simple_preprocess

import joblib

from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords


# Load data from drive
from google.colab import drive

drive.mount('/content/drive')

data = pd.read_csv(r'/content/drive/MyDrive/all_recipe.csv')


# Data Cleaning (remove dummy words)
recipe = data['title'].astype(str) + ' ' + data['ingredients'].astype(str) + ' ' + data['instructions'].astype(str)

recipe = [re.sub(r'\b\w*ADVERTISEMENT\w*\b', '', sentence) for sentence in recipe]

recipe = [''.join(words) for words in recipe]

recipe = pd.Series(recipe)


# Extend stopwords
import nltk

from nltk.corpus import stopwords

nltk.download('stopwords')

english_stopwords = stopwords.words('english')

english_stopwords.append('make')


# TFIDF settings and apply it to the dataset
tfidf = TfidfVectorizer(lowercase=True, stop_words='english', min_df=0.05, max_df=0.9, ngram_range = (1,3))

tfidf.fit(recipe)

recipe_data = tfidf.fit_transform(recipe)


# Use cosine similarity to compare the TFIDF-transformed data with the TFIDF-transformed user query (print out the top 3 related recipes)
question = input("What is in your mind? \n")

X = tfidf.transform([question])

cos_X = cosine_similarity(X, recipe_data)

top_indices = np.argsort(cos_X[0])[-3:][::-1]

print(question)

print('---------------------------------------------------------------------------')

for i, index in enumerate(top_indices, 1):

    recipe_info = re.split('\[|\]', recipe[index])
    print(f"Recommendation {i} , The similarity is {cos_X[0][index]}:")
    print('*Title:')
    print(recipe_info[0],'\n')
    print('*Ingredients:')
    print(recipe_info[1],'\n')
    print('*Instructions:')
    print(recipe_info[2],'\n')
    print('\n')


# Try TFIDF+SVD (result seems to be worse)
svd = TruncatedSVD(10)

R = svd.fit_transform(recipe_data)

Q = svd.transform(X)

cos_X = cosine_similarity(Q, R)

top_indices = np.argsort(cos_X[0])[-3:][::-1]

print(question)

print('---------------------------------------------------------------------------')

for i, index in enumerate(top_indices, 1):

    recipe_info = re.split('\[|\]', recipe[index])
    print(f"Recommendation {i} , The similarity is {cos_X[0][index]}:")
    print('*Title:')
    print(recipe_info[0],'\n')
    print('*Ingredients:')
    print(recipe_info[1],'\n')
    print('*Instructions:')
    print(recipe_info[2],'\n')
    print('\n')


# Word2Vec settings and apply it to the dataset
toks = []

for s in recipe:

  toks.append(simple_preprocess(s))

model = Word2Vec(toks, vector_size=400, window=5, min_count=0, workers=4)

word_vectors = model.wv

# Data Cleaning
question_tokens = simple_preprocess(question)

question_vec = word_vectors[question_tokens]

question_avg_vec = np.mean([word_vectors[token] for token in question_tokens if token in word_vectors], axis=0)

question_avg_vec = np.nan_to_num(question_avg_vec)

question_avg_vec /= np.linalg.norm(question_avg_vec)


# Compare the Word2Vec-transformed data with the Word2Vec-transformed user query (print out the top 3 related recipes) (the result seems terrible)
print(question)

print('---------------------------------------------------------------------------')

recipe = data['title'].astype(str) + ' ' + data['ingredients'].astype(str) + ' ' + data['instructions'].astype(str)

recipe_similarity = []

for content in recipe:

    content_tokens = simple_preprocess(content)
    recipe_avg_vec = np.mean([word_vectors[token] for token in content_tokens if token in word_vectors], axis=0)
    recipe_avg_vec = np.nan_to_num(recipe_avg_vec)
    recipe_avg_vec /= np.linalg.norm(recipe_avg_vec)
    recipe_similarity.append(np.dot(question_avg_vec, recipe_avg_vec))

top3_indices = np.argsort(recipe_similarity)[::-1][:3]

for i, index in enumerate(top3_indices, 1):

    recipe_info = re.split('\[|\]', recipe[index])
    print(f"Recommendation {i} , The similarity is {recipe_similarity[index]}:")
    print('*Title:')
    print(recipe_info[0],'\n')
    print('*Ingredients:')
    print(recipe_info[1],'\n')
    print('*Instructions:')
    print(recipe_info[2],'\n')
    print('\n')


# Word2Vec + SVD (the result seems to be worse and worse)
svd = TruncatedSVD(10)

svd.fit(word_vectors.vectors)

word_vectors_svd = svd.transform(word_vectors.vectors)

question_avg_vec = np.mean([word_vectors_svd[word_vectors.index_to_key.index(token)] for token in question_tokens if token in word_vectors], axis=0)

question_avg_vec = np.nan_to_num(question_avg_vec)

question_avg_vec /= np.linalg.norm(question_avg_vec)

print(question)

print('---------------------------------------------------------------------------')

recipe_similarity = []

for content in recipe:

    content_tokens = simple_preprocess(content)
    recipe_avg_vec = np.mean([word_vectors_svd[word_vectors.index_to_key.index(token)] for token in content_tokens if token in word_vectors], axis=0)
    recipe_avg_vec = np.nan_to_num(recipe_avg_vec)
    recipe_avg_vec /= np.linalg.norm(recipe_avg_vec)
    recipe_similarity.append(np.dot(question_avg_vec, recipe_avg_vec))

top3_indices = np.argsort(recipe_similarity)[::-1][:3]

for i, index in enumerate(top3_indices, 1):

    recipe_info = re.split('\[|\]', recipe[index])
    print(f"Recommendation {i} , The similarity is {recipe_similarity[index]}:")
    print('*Title:')
    print(recipe_info[0],'\n')
    print('*Ingredients:')
    print(recipe_info[1],'\n')
    print('*Instructions:')
    print(recipe_info[2],'\n')
    print('\n')


# CountVectorizer (The result seems to be as good as TFIDF)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b', stop_words='english')

recipe_bow = vectorizer.fit_transform(recipe)

question_bow = vectorizer.transform([question])

recipe_intersections = recipe_bow.multiply(question_bow).sum(axis=1)

recipe_unions = recipe_bow.sum(axis=1) + question_bow.sum(axis=1) - recipe_intersections

jaccard_similarity = recipe_intersections / recipe_unions

top_indices = jaccard_similarity.argsort(axis=0)[-3:][::-1]

top_indices = np.ravel(top_indices)

print(question)

print('---------------------------------------------------------------------------')

for i, index in enumerate(top_indices, 1):

    recipe_info = re.split('\[|\]', recipe[index])
    print(f"Recommendation {i} , The similarity is {jaccard_similarity[index][0]}:")
    print('*Title:')
    print(recipe_info[0],'\n')
    print('*Ingredients:')
    print(recipe_info[1],'\n')
    print('*Instructions:')
    print(recipe_info[2],'\n')
    print('\n')


# Combine TFIDF with CountVectorizer to a mixed model
cos_X = np.ravel(cos_X)

jaccard_similarity = np.ravel(jaccard_similarity)

similarity = cos_X + jaccard_similarity

top_indices = similarity.argsort(axis=0)[-3:]

print(top_indices)

print(question)

print('---------------------------------------------------------------------------')

for i, index in enumerate(top_indices, 1):

    recipe_info = re.split('\[|\]', recipe[index])
    print(f"Recommendation {i} , The similarity is {similarity[index]}:")
    print('*Title:')
    print(recipe_info[0],'\n')
    print('*Ingredients:')
    print(recipe_info[1],'\n')
    print('*Instructions:')
    print(recipe_info[2],'\n')
    print('\n')


# Try to categorize the recipe dataset into different cuisines and then build specific models

recipe1 = recipe.apply(lambda x: x.split())

indian_cuisine = ['Indian']

mexican_cuisine = ['Mexican']

french_cuisine = ['French']

italian_cuisine = ['Italian']

japanese_cuisine = ['Japanese']

korean_cuisine = ['Korean']

spanish_cuisine = ['Spanish']

thai_cuisine = ['Thai']

american_cuisine = ['American']

chinese_cuisine = ['Chinese']

recipe_indian = []

recipe_mexican = []

recipe_french = []

recipe_italian = []

recipe_japanese = []

recipe_korean = []

recipe_spanish = []

recipe_thai = []

recipe_american = []

recipe_chinese = []

for index,word_list in enumerate(recipe1):

    lowercase_word_list = [word.lower() for word in word_list]

    if any(keyword.lower() in lowercase_word_list for keyword in indian_cuisine):
        recipe_indian.append(recipe[index])
    elif any(keyword.lower() in lowercase_word_list for keyword in mexican_cuisine):
        recipe_mexican.append(recipe[index])
    elif any(keyword.lower() in lowercase_word_list for keyword in french_cuisine):
        recipe_french.append(recipe[index])
    elif any(keyword.lower() in lowercase_word_list for keyword in italian_cuisine):
        recipe_italian.append(recipe[index])
    elif any(keyword.lower() in lowercase_word_list for keyword in japanese_cuisine):
        recipe_japanese.append(recipe[index])
    elif any(keyword.lower() in lowercase_word_list for keyword in korean_cuisine):
        recipe_korean.append(recipe[index])
    elif any(keyword.lower() in lowercase_word_list for keyword in spanish_cuisine):
        recipe_spanish.append(recipe[index])
    elif any(keyword.lower() in lowercase_word_list for keyword in thai_cuisine):
        recipe_thai.append(recipe[index])
    elif any(keyword.lower() in lowercase_word_list for keyword in american_cuisine):
        recipe_american.append(recipe[index])
    elif any(keyword.lower() in lowercase_word_list for keyword in chinese_cuisine):
        recipe_chinese.append(recipe[index])

recipe_indian = pd.Series(recipe_indian)

recipe_mexican = pd.Series(recipe_mexican)

recipe_french = pd.Series(recipe_french)

recipe_italian = pd.Series(recipe_italian)

recipe_japanese = pd.Series(recipe_japanese)

recipe_korean = pd.Series(recipe_korean)

recipe_spanish = pd.Series(recipe_spanish)

recipe_thai = pd.Series(recipe_thai)

recipe_american = pd.Series(recipe_american)

recipe_chinese = pd.Series(recipe_chinese)

recipe_indian.to_csv('/content/drive/My Drive/Recipe Recommendation/recipe_indian.csv', index=True)

recipe_mexican.to_csv('/content/drive/My Drive/Recipe Recommendation/recipe_mexican.csv', index=True)

recipe_french.to_csv('/content/drive/My Drive/Recipe Recommendation/recipe_french.csv', index=True)

recipe_italian.to_csv('/content/drive/My Drive/Recipe Recommendation/recipe_italian.csv', index=True)

recipe_japanese.to_csv('/content/drive/My Drive/Recipe Recommendation/recipe_japanese.csv', index=True)

recipe_korean.to_csv('/content/drive/My Drive/Recipe Recommendation/recipe_korean.csv', index=True)

recipe_spanish.to_csv('/content/drive/My Drive/Recipe Recommendation/recipe_spanish.csv', index=True)

recipe_thai.to_csv('/content/drive/My Drive/Recipe Recommendation/recipe_thai.csv', index=True)

recipe_american.to_csv('/content/drive/My Drive/Recipe Recommendation/recipe_american.csv', index=True)

recipe_chinese.to_csv('/content/drive/My Drive/Recipe Recommendation/recipe_chinese.csv', index=True)

  # 1
tfidf_ind = TfidfVectorizer(lowercase=True, stop_words=english_stopwords, min_df=0.05, max_df=0.9, ngram_range = (1,3))

vectorizer_ind = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b', stop_words=english_stopwords)

tfidf_ind.fit(recipe_indian)

recipe_ind = tfidf_ind.fit_transform(recipe_indian)

vectorizer_ind.fit(recipe_indian)

recipe_IND = vectorizer_ind.fit_transform(recipe_indian)

joblib.dump(tfidf_ind, '/content/drive/My Drive/Recipe Recommendation/tfidf_ind.pkl')

joblib.dump(vectorizer_ind, '/content/drive/My Drive/Recipe Recommendation/vectorizer_ind.pkl')

joblib.dump(recipe_ind, '/content/drive/My Drive/Recipe Recommendation/recipe_ind.pkl')

joblib.dump(recipe_IND, '/content/drive/My Drive/Recipe Recommendation/recipe_IND.pkl')

  # 2
tfidf_mex = TfidfVectorizer(lowercase=True, stop_words=english_stopwords, min_df=0.05, max_df=0.9, ngram_range = (1,3))

vectorizer_mex = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b', stop_words=english_stopwords)

tfidf_mex.fit(recipe_mexican)

vectorizer_mex.fit(recipe_mexican)

recipe_mex = tfidf_mex.fit_transform(recipe_mexican)

recipe_MEX = vectorizer_mex.fit_transform(recipe_mexican)

joblib.dump(tfidf_mex, '/content/drive/My Drive/Recipe Recommendation/tfidf_mex.pkl')

joblib.dump(vectorizer_mex, '/content/drive/My Drive/Recipe Recommendation/vectorizer_mex.pkl')

joblib.dump(recipe_mex, '/content/drive/My Drive/Recipe Recommendation/recipe_mex.pkl')

joblib.dump(recipe_MEX, '/content/drive/My Drive/Recipe Recommendation/recipe_MEX.pkl')

  # 3
tfidf_fre = TfidfVectorizer(lowercase=True, stop_words=english_stopwords, min_df=0.05, max_df=0.9, ngram_range = (1,3))

vectorizer_fre = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b', stop_words=english_stopwords)

tfidf_fre.fit(recipe_french)

vectorizer_fre.fit(recipe_french)

recipe_fre = tfidf_fre.fit_transform(recipe_french)

recipe_FRE = vectorizer_fre.fit_transform(recipe_french)

joblib.dump(tfidf_fre, '/content/drive/My Drive/Recipe Recommendation/tfidf_fre.pkl')

joblib.dump(vectorizer_fre, '/content/drive/My Drive/Recipe Recommendation/vectorizer_fre.pkl')

joblib.dump(recipe_fre, '/content/drive/My Drive/Recipe Recommendation/recipe_fre.pkl')

joblib.dump(recipe_FRE, '/content/drive/My Drive/Recipe Recommendation/recipe_FRE.pkl')

  # 4
tfidf_ita = TfidfVectorizer(lowercase=True, stop_words=english_stopwords, min_df=0.05, max_df=0.9, ngram_range = (1,3))

vectorizer_ita = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b', stop_words=english_stopwords)

tfidf_ita.fit(recipe_italian)

vectorizer_ita.fit(recipe_italian)

recipe_ita = tfidf_ita.fit_transform(recipe_italian)

recipe_ITA = vectorizer_ita.fit_transform(recipe_italian)

joblib.dump(tfidf_ita, '/content/drive/My Drive/Recipe Recommendation/tfidf_ita.pkl')

joblib.dump(vectorizer_ita, '/content/drive/My Drive/Recipe Recommendation/vectorizer_ita.pkl')

joblib.dump(recipe_ita, '/content/drive/My Drive/Recipe Recommendation/recipe_ita.pkl')

joblib.dump(recipe_ITA, '/content/drive/My Drive/Recipe Recommendation/recipe_ITA.pkl')

  # 5
tfidf_jap = TfidfVectorizer(lowercase=True, stop_words=english_stopwords, min_df=0.05, max_df=0.9, ngram_range = (1,3))

vectorizer_jap = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b', stop_words=english_stopwords)

tfidf_jap.fit(recipe_japanese)

vectorizer_jap.fit(recipe_japanese)

recipe_jap = tfidf_jap.fit_transform(recipe_japanese)

recipe_JAP = vectorizer_jap.fit_transform(recipe_japanese)

joblib.dump(tfidf_jap, '/content/drive/My Drive/Recipe Recommendation/tfidf_jap.pkl')

joblib.dump(vectorizer_jap, '/content/drive/My Drive/Recipe Recommendation/vectorizer_jap.pkl')

joblib.dump(recipe_jap, '/content/drive/My Drive/Recipe Recommendation/recipe_jap.pkl')

joblib.dump(recipe_JAP, '/content/drive/My Drive/Recipe Recommendation/recipe_JAP.pkl')

  # 6
tfidf_kor = TfidfVectorizer(lowercase=True, stop_words=english_stopwords, min_df=0.05, max_df=0.9, ngram_range = (1,3))

vectorizer_kor = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b', stop_words=english_stopwords)

tfidf_kor.fit(recipe_korean)

vectorizer_kor.fit(recipe_korean)

recipe_kor = tfidf_kor.fit_transform(recipe_korean)

recipe_KOR = vectorizer_kor.fit_transform(recipe_korean)

joblib.dump(tfidf_kor, '/content/drive/My Drive/Recipe Recommendation/tfidf_kor.pkl')

joblib.dump(vectorizer_kor, '/content/drive/My Drive/Recipe Recommendation/vectorizer_kor.pkl')

joblib.dump(recipe_kor, '/content/drive/My Drive/Recipe Recommendation/recipe_kor.pkl')

joblib.dump(recipe_KOR, '/content/drive/My Drive/Recipe Recommendation/recipe_KOR.pkl')

  # 7
tfidf_spa = TfidfVectorizer(lowercase=True, stop_words=english_stopwords, min_df=0.05, max_df=0.9, ngram_range = (1,3))

vectorizer_spa = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b', stop_words=english_stopwords)

tfidf_spa.fit(recipe_spanish)

vectorizer_spa.fit(recipe_spanish)

recipe_spa = tfidf_spa.fit_transform(recipe_spanish)

recipe_SPA = vectorizer_spa.fit_transform(recipe_spanish)

joblib.dump(tfidf_spa, '/content/drive/My Drive/Recipe Recommendation/tfidf_spa.pkl')

joblib.dump(vectorizer_spa, '/content/drive/My Drive/Recipe Recommendation/vectorizer_spa.pkl')

joblib.dump(recipe_spa, '/content/drive/My Drive/Recipe Recommendation/recipe_spa.pkl')

joblib.dump(recipe_SPA, '/content/drive/My Drive/Recipe Recommendation/recipe_SPA.pkl')

  # 8
tfidf_tha = TfidfVectorizer(lowercase=True, stop_words=english_stopwords, min_df=0.05, max_df=0.9, ngram_range = (1,3))

vectorizer_tha = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b', stop_words=english_stopwords)

tfidf_tha.fit(recipe_thai)

vectorizer_tha.fit(recipe_thai)

recipe_tha = tfidf_tha.fit_transform(recipe_thai)

recipe_THA = vectorizer_tha.fit_transform(recipe_thai)

joblib.dump(tfidf_tha, '/content/drive/My Drive/Recipe Recommendation/tfidf_tha.pkl')

joblib.dump(vectorizer_tha, '/content/drive/My Drive/Recipe Recommendation/vectorizer_tha.pkl')

joblib.dump(recipe_tha, '/content/drive/My Drive/Recipe Recommendation/recipe_tha.pkl')

joblib.dump(recipe_THA, '/content/drive/My Drive/Recipe Recommendation/recipe_THA.pkl')

  # 9
tfidf_ame = TfidfVectorizer(lowercase=True, stop_words=english_stopwords, min_df=0.05, max_df=0.9, ngram_range = (1,3))

vectorizer_ame = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b', stop_words=english_stopwords)

tfidf_ame.fit(recipe_american)

vectorizer_ame.fit(recipe_american)

recipe_ame = tfidf_ame.fit_transform(recipe_american)

recipe_AME = vectorizer_ame.fit_transform(recipe_american)

joblib.dump(tfidf_ame, '/content/drive/My Drive/Recipe Recommendation/tfidf_ame.pkl')

joblib.dump(vectorizer_ame, '/content/drive/My Drive/Recipe Recommendation/vectorizer_ame.pkl')

joblib.dump(recipe_ame, '/content/drive/My Drive/Recipe Recommendation/recipe_ame.pkl')

joblib.dump(recipe_AME, '/content/drive/My Drive/Recipe Recommendation/recipe_AME.pkl')

  # 10
tfidf_chi = TfidfVectorizer(lowercase=True, stop_words=english_stopwords, min_df=0.05, max_df=0.9, ngram_range = (1,3))

vectorizer_chi = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b', stop_words=english_stopwords)

tfidf_chi.fit(recipe_chinese)

vectorizer_chi.fit(recipe_chinese)

recipe_chi = tfidf_chi.fit_transform(recipe_chinese)

recipe_CHI = vectorizer_chi.fit_transform(recipe_chinese)

joblib.dump(tfidf_chi, '/content/drive/My Drive/Recipe Recommendation/tfidf_chi.pkl')

joblib.dump(vectorizer_chi, '/content/drive/My Drive/Recipe Recommendation/vectorizer_chi.pkl')

joblib.dump(recipe_chi, '/content/drive/My Drive/Recipe Recommendation/recipe_chi.pkl')

joblib.dump(recipe_CHI, '/content/drive/My Drive/Recipe Recommendation/recipe_CHI.pkl')

  # 11
tfidf_all = TfidfVectorizer(lowercase=True, stop_words=english_stopwords, min_df=0.05, max_df=0.9, ngram_range = (1,3))

vectorizer_all = CountVectorizer(lowercase=True, token_pattern=r'(?u)\b\w+\b', stop_words=english_stopwords)

tfidf_all.fit(recipe)

vectorizer_all.fit(recipe)

recipe_all = tfidf_all.fit_transform(recipe)

recipe_ALL = vectorizer_all.fit_transform(recipe)

joblib.dump(tfidf_all, '/content/drive/My Drive/Recipe Recommendation/tfidf_all.pkl')

joblib.dump(vectorizer_all, '/content/drive/My Drive/Recipe Recommendation/vectorizer_all.pkl')

joblib.dump(recipe_all, '/content/drive/My Drive/Recipe Recommendation/recipe_all.pkl')

joblib.dump(recipe_ALL, '/content/drive/My Drive/Recipe Recommendation/recipe_ALL.pkl')


# Read the stored model and categorized recipe data from drive
tfidf_indian = joblib.load('/content/drive/My Drive/Recipe Recommendation/tfidf_ind.pkl')

vectorizer_indian = joblib.load('/content/drive/My Drive/Recipe Recommendation/vectorizer_ind.pkl')

recipe_ind = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_ind.pkl')

recipe_IND = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_IND.pkl')

tfidf_mexican = joblib.load('/content/drive/My Drive/Recipe Recommendation/tfidf_mex.pkl')

vectorizer_mexican = joblib.load('/content/drive/My Drive/Recipe Recommendation/vectorizer_mex.pkl')

recipe_mex = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_mex.pkl')

recipe_MEX = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_MEX.pkl')

tfidf_french = joblib.load('/content/drive/My Drive/Recipe Recommendation/tfidf_fre.pkl')

vectorizer_french = joblib.load('/content/drive/My Drive/Recipe Recommendation/vectorizer_fre.pkl')

recipe_fre = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_fre.pkl')

recipe_FRE = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_FRE.pkl')

tfidf_italian = joblib.load('/content/drive/My Drive/Recipe Recommendation/tfidf_ita.pkl')

vectorizer_italian = joblib.load('/content/drive/My Drive/Recipe Recommendation/vectorizer_ita.pkl')

recipe_ita = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_ita.pkl')

recipe_ITA = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_ITA.pkl')

tfidf_japanese = joblib.load('/content/drive/My Drive/Recipe Recommendation/tfidf_jap.pkl')

vectorizer_japanese = joblib.load('/content/drive/My Drive/Recipe Recommendation/vectorizer_jap.pkl')

recipe_jap = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_jap.pkl')

recipe_JAP = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_JAP.pkl')

tfidf_korean = joblib.load('/content/drive/My Drive/Recipe Recommendation/tfidf_kor.pkl')

vectorizer_korean = joblib.load('/content/drive/My Drive/Recipe Recommendation/vectorizer_kor.pkl')

recipe_kor = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_kor.pkl')

recipe_KOR = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_KOR.pkl')

tfidf_spanish = joblib.load('/content/drive/My Drive/Recipe Recommendation/tfidf_spa.pkl')

vectorizer_spanish = joblib.load('/content/drive/My Drive/Recipe Recommendation/vectorizer_spa.pkl')

recipe_spa = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_spa.pkl')

recipe_SPA = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_SPA.pkl')

tfidf_thai = joblib.load('/content/drive/My Drive/Recipe Recommendation/tfidf_tha.pkl')

vectorizer_thai = joblib.load('/content/drive/My Drive/Recipe Recommendation/vectorizer_tha.pkl')

recipe_tha = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_tha.pkl')

recipe_THA = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_THA.pkl')

tfidf_american = joblib.load('/content/drive/My Drive/Recipe Recommendation/tfidf_ame.pkl')

vectorizer_american = joblib.load('/content/drive/My Drive/Recipe Recommendation/vectorizer_ame.pkl')

recipe_ame = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_ame.pkl')

recipe_AME = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_AME.pkl')

tfidf_chinese = joblib.load('/content/drive/My Drive/Recipe Recommendation/tfidf_chi.pkl')

vectorizer_chinese = joblib.load('/content/drive/My Drive/Recipe Recommendation/vectorizer_chi.pkl')

recipe_chi = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_chi.pkl')

recipe_CHI = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_CHI.pkl')

tfidf = joblib.load('/content/drive/My Drive/Recipe Recommendation/tfidf_all.pkl')

vectorizer = joblib.load('/content/drive/My Drive/Recipe Recommendation/vectorizer_all.pkl')

recipe_all = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_all.pkl')

recipe_ALL = joblib.load('/content/drive/My Drive/Recipe Recommendation/recipe_ALL.pkl')

recipe_indian = pd.read_csv('/content/drive/MyDrive/Recipe Recommendation/recipe_indian.csv')

recipe_mexican = pd.read_csv('/content/drive/MyDrive/Recipe Recommendation/recipe_mexican.csv')

recipe_french = pd.read_csv('/content/drive/MyDrive/Recipe Recommendation/recipe_french.csv')

recipe_italian = pd.read_csv('/content/drive/MyDrive/Recipe Recommendation/recipe_italian.csv')

recipe_japanese = pd.read_csv('/content/drive/MyDrive/Recipe Recommendation/recipe_japanese.csv')

recipe_korean = pd.read_csv('/content/drive/MyDrive/Recipe Recommendation/recipe_korean.csv')

recipe_spanish = pd.read_csv('/content/drive/MyDrive/Recipe Recommendation/recipe_spanish.csv')

recipe_thai = pd.read_csv('/content/drive/MyDrive/Recipe Recommendation/recipe_thai.csv')

recipe_american = pd.read_csv('/content/drive/MyDrive/Recipe Recommendation/recipe_american.csv')

recipe_chinese = pd.read_csv('/content/drive/MyDrive/Recipe Recommendation/recipe_chinese.csv')


# Create a dictionary for comparing the users' query and the specific recipes
cuisines = {

    "indian_cuisine": {
        "list": indian_cuisine,
        "tfidf": tfidf_indian,
        "vect": vectorizer_indian,
        "transform": recipe_ind,
        "transform1": recipe_IND,
        "dataset": recipe_indian
    },
    "mexican_cuisine": {
        "list": mexican_cuisine,
        "tfidf": tfidf_mexican,
        "vect": vectorizer_mexican,
        "transform": recipe_mex,
        "transform1": recipe_MEX,
        "dataset": recipe_mexican
    },
    "french_cuisine": {
        "list": french_cuisine,
        "tfidf": tfidf_french,
        "vect": vectorizer_french,
        "transform": recipe_fre,
        "transform1": recipe_FRE,
        "dataset": recipe_french
    },
    "italian_cuisine": {
        "list": italian_cuisine,
        "tfidf": tfidf_italian,
        "vect": vectorizer_italian,
        "transform": recipe_ita,
        "transform1": recipe_ITA,
        "dataset": recipe_italian
    },
    "japanese_cuisine": {
        "list": japanese_cuisine,
        "tfidf": tfidf_japanese,
        "vect": vectorizer_japanese,
        "transform": recipe_jap,
        "transform1": recipe_JAP,
        "dataset": recipe_japanese
    },
    "korean_cuisine": {
        "list": korean_cuisine,
        "tfidf": tfidf_korean,
        "vect": vectorizer_korean,
        "transform": recipe_kor,
        "transform1": recipe_KOR,
        "dataset": recipe_korean
    },
    "spanish_cuisine": {
        "list": spanish_cuisine,
        "tfidf": tfidf_spanish,
        "vect": vectorizer_spanish,
        "transform": recipe_spa,
        "transform1": recipe_SPA,
        "dataset": recipe_spanish
    },
    "thai_cuisine": {
        "list": thai_cuisine,
        "tfidf": tfidf_thai,
        "vect": vectorizer_thai,
        "transform": recipe_tha,
        "transform1": recipe_THA,
        "dataset": recipe_thai
    },
    "american_cuisine": {
        "list": american_cuisine,
        "tfidf": tfidf_american,
        "vect": vectorizer_american,
        "transform": recipe_ame,
        "transform1": recipe_AME,
        "dataset": recipe_american
    },
    "chinese_cuisine": {
        "list": chinese_cuisine,
        "tfidf": tfidf_chinese,
        "vect": vectorizer_chinese,
        "transform": recipe_chi,
        "transform1": recipe_CHI,
        "dataset": recipe_chinese
    },
}


# Use the mixed approach to find the top 3 related recipes

question = input("What is in your mind?\n")

question_lower = question.lower()

for typ, cuisine in cuisines.items():

    for key_word in cuisine["list"]:
        if key_word.lower() in question_lower:
            X = cuisine["tfidf"].transform([question])
            Y = cuisine["vect"].transform([question])
            cosine_sim = np.ravel(cosine_similarity(X, cuisine["transform"]))
            recipe_intersections = cuisine["transform1"].multiply(Y).sum(axis=1)
            recipe_unions = cuisine["transform1"].sum(axis=1) + Y.sum(axis=1) - recipe_intersections
            jaccard_similarity = recipe_intersections / recipe_unions
            jaccard_sim = np.ravel(jaccard_similarity)
            sim = cosine_sim + jaccard_sim
            top_indices = np.argsort(sim)[-3:]
            print(question)
            print('---------------------------------------------------------------------------')
            for i, index in enumerate(top_indices, 1):
                recipe_info = re.split('\[|\]', cuisine["dataset"][index])
                print(f"Recommendation {i} , The similarity is {sim[index]}:")
                print('*Title:')
                print(recipe_info[0],'\n')
                print('*Ingredients:')
                print(recipe_info[1],'\n')
                print('*Instructions:')
                print(recipe_info[2],'\n')
                print('\n')
            break
    else:
        continue
    break

else:

    X = tfidf.transform([question])
    Y = vectorizer.transform([question])
    cosine_sim = np.ravel(cosine_similarity(X, recipe_all))
    recipe_intersections = recipe_ALL.multiply(Y).sum(axis=1)
    recipe_unions = recipe_ALL.sum(axis=1) + Y.sum(axis=1) - recipe_intersections
    jaccard_similarity = recipe_intersections / recipe_unions
    jaccard_sim = np.ravel(jaccard_similarity)
    sim = cosine_sim + jaccard_sim
    top_indices = np.argsort(sim)[-3:]
    print(question)
    print('---------------------------------------------------------------------------')
    for i, index in enumerate(top_indices, 1):
        recipe_info = re.split('\[|\]', recipe[index])
        print(f"Recommendation {i} , The similarity is {sim[index]}:")
        print('*Title:')
        print(recipe_info[0],'\n')
        print('*Ingredients:')
        print(recipe_info[1],'\n')
        print('*Instructions:')
        print(recipe_info[2],'\n')
        print('\n')
