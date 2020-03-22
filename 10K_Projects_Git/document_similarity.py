# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import nltk
import string
import requests
import re
import glob
import os, sys
import datetime
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import xlrd
import csv
import time
import dask
import dask.dataframe as dd
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
from sklearn.feature_extraction.text import TfidfVectorizer
import shutil
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
import nltk

nltk.download('stopwords')
dl = Downloader("/home/hongzhuoqiao/10K_Projects/sec_filings")


# Similarity Function with Preprocessing
# doc_test_list =['  <Hello% my_World 2020! BYE    2019@!!   ', 'item.1A HELLO the WORLD 2019.']
# docs means doc_list
def preprocess(docs):
    sepcial_stopwords =['item','items','itemsa'] # create sepcial stopwords set
    docs = [doc.lower() for doc in docs] # lowercase
    table = str.maketrans('','', string.punctuation) # remove punctuation 1
    docs = [doc.translate(table) for doc in docs] # remove punctuation 2
    docs = [re.sub(r'\d+', 'num', doc) for doc in docs] # replace all # to 'num'
    docs = [re.sub('[^\w\s]', '',doc) for doc in docs] # remove special char
    docs = [re.sub('_', '', doc) for doc in docs] # remove underscore
    docs = [re.sub('\s+',' ', doc) for doc in docs] # change whitespaces to one space
    stopwords = set(nltk.corpus.stopwords.words('english')+sepcial_stopwords) # create stopwords set with special stopword if have any 1
    docs = [[word for word in doc.split() if word not in stopwords] for doc in docs]  # create stopwords set with special stopword if have any 2
    stemmer = nltk.stem.PorterStemmer() # stemming 1
    docs = [" ".join([stemmer.stem(word) for word in doc]) for doc in docs] # stemming 2
    docs = [doc.strip() for doc in docs]
    return docs

def similarity_calculation(ticker):
    print("\n>>> Start meausuring company {}".format(ticker))
    path_header = "/home/hongzhuoqiao/10K_Projects/sec_filings/item1_section"
    path_for_company_folder = path_header + "/" + ticker +"/*.txt"
    files = glob.glob(path_for_company_folder)
    file_list =[]
    filename_list =[]
    for file in files:
        f = open(file, 'r')
        filename = os.path.basename(file)
        file_list.append(f.read())
        filename_list.append(filename[-6:-4])

    filename_list = [datetime.datetime.strptime(f,'%y').strftime('%Y') for f in filename_list]
    file_list = preprocess(file_list)
    tfidf = TfidfVectorizer(stop_words='english', strip_accents='ascii').fit_transform(file_list)
    pairwise_similarity = tfidf*tfidf.T
    sim_df = pd.DataFrame(pairwise_similarity.toarray(), columns = filename_list, index = filename_list)
    sim_df.sort_index(axis=0, ascending=False, inplace=True)
    sim_df.sort_index(axis=1, ascending=False, inplace=True)
    #sim_df.to_csv("/home/hongzhuoqiao/10K_Projects/sec_filings/tfidf_similarity/sim_matrix/"+ticker+"_matrix.csv")
    print (">>>) Got similarity matrix for {}".format(ticker))

    return sim_df

#similarity_calculation(ticker).to_csv('apple_sim_example.csv')

# %%
# Last_3y_sim calculation from existing sim_matrix


def last_3y_similarity(sim_df,tickername):
    last_3y_sim_list =[]
    for i in range(0,len(sim_df)-3):
        last_3y_sim_list.append((np.sum(sim_df.iloc[i,i+1] + sim_df.iloc[i,i+2] + sim_df.iloc[i,i+3]))/3)
    last_3y_sim_df = pd.DataFrame(last_3y_sim_list, index = sim_df.columns[:-3],columns=[tickername])
    return last_3y_sim_df


# %%
# Last_5y_sim calculation from existing sim_matrix
def last_5y_similarity(sim_df,tickername):
    last_5y_sim_list =[]
    for i in range(0,len(sim_df)-5):
        last_5y_sim_list.append((np.sum(sim_df.iloc[i,i+1] + sim_df.iloc[i,i+2] + sim_df.iloc[i,i+3] + sim_df.iloc[i,i+4] + sim_df.iloc[i,i+5]))/5)
    last_5y_sim_df = pd.DataFrame(last_5y_sim_list, index = sim_df.columns[:-5],columns=[tickername])
    return last_5y_sim_df
    