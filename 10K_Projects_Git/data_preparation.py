
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
dl = Downloader("/home/hongzhuoqiao/10K_Projects/sec_filings")


# %%
def clean_ticker(tic):
    
    if re.match('\w+[.]\w+', str(tic)):
        if re.match('\w+[.]\d+', str(tic)):
            tic = ''.join([i for i in tic if not i.isdigit()])
            tic = ''.join([i for i in tic if not i is "."])    
        else:
            tic = ''.join([i for i in tic if not i is "."])

    if re.match('\w+[.]', str(tic)):        
        tic = ''.join([i for i in tic if not i is "."]) 
    return tic

# %%
## main function of extraction items for all companies
def download_to_item1(ticker):
    file_number = dl.get("10-K", ticker)
    print ('>>> Got {} for {}'.format(file_number, ticker))
    error = 0
    count = 1
    extract_failed_flag = 0
    error_file_list = []

    if file_number:

        path_header = "/home/hongzhuoqiao/10K_Projects/sec_filings/sec_edgar_filings"
        path_for_extraction_item1 = path_header + "/" + ticker + "/10-K" +"/*.txt"
        path_for_item1_folder = "/home/hongzhuoqiao/10K_Projects/sec_filings/item1_section/" + ticker
        if not os.path.exists(path_for_item1_folder):
            os.makedirs(path_for_item1_folder)
            print (">>> Created new folder for {}".format(ticker))
        else:
            print (">>> {} folder is already exists.".format(ticker))

        print (">>> Start extracting item1 sections for {}".format(ticker))
    
        files = glob.glob(path_for_extraction_item1)

        # extract annual file for one company, loop every year in one company
       
        for file in files:
            f = open(file, 'r')
            print(" \n\n>>>Starting open files {} for {}".format(count, ticker))
            filename = os.path.basename(file)
            print (">>> the file to be extracted is {}\n".format(filename))
            newname_suffix = filename[10:13]
            file_name = ticker+newname_suffix
            
            raw_10k = f.read()    
            ######
            # Regex to find <DOCUMENT> tags
        
            doc_start_pattern = re.compile(r'<DOCUMENT>')
            doc_end_pattern = re.compile(r'</DOCUMENT>')
            # Regex to find <TYPE> tag prceeding any characters, terminating at new line
            type_pattern = re.compile(r'<TYPE>[^\n]+')

            ### as section names
            doc_start_is = [x.end() for x in doc_start_pattern.finditer(raw_10k)]
            doc_end_is = [x.start() for x in doc_end_pattern.finditer(raw_10k)]
            doc_types = [x[len('<TYPE>'):] for x in type_pattern.findall(raw_10k)]

            #regex = re.compile(r'(Item(\s|&#160;|&nbsp;)(1A|1B|1|2)\.{0,1})|(>Item(\s|&#160;|&nbsp;)(1A|1B|1|2)\.{0,1})|(ITEM(\s|&#160;|&nbsp;)(1A|1B|1|2).{0,1})')

            regex = re.compile(r'(Item(\s|&#160;|&nbsp;)(1A|1B|1|2)\.{0})|(>Item(\s|&#160;|&nbsp;)(1A|1B|1|2)\.{0})|(ITEM(\s|&#160;|&nbsp;)(1A|1B|1|2)\.{0})')

            # Create a loop to go through each section type and save only the 10-K section in the dictionary
            document = {}
            doc = ''
            for doc_type, doc_start, doc_end in zip(doc_types, doc_start_is, doc_end_is):
                
                if (doc_type =='10-K'):
                    document[doc_type] = raw_10k[doc_start:doc_end]    
                    # Use finditer to math the regex
                    matches = regex.finditer(document['10-K'])
                    doc = '10-K'                    
                    # Write a for loop to print the matches                   
                elif (doc_type =='10-K405'):
                    document[doc_type] = raw_10k[doc_start:doc_end]    
                    # Use finditer to math the regex
                    matches = regex.finditer(document['10-K405'])
                    doc = '10-K405'

                elif (doc_type =='10-KT'):
                    document[doc_type] = raw_10k[doc_start:doc_end]    
                    # Use finditer to math the regex
                    matches = regex.finditer(document['10-KT'])
                    doc = '10-KT'

            
            # Create the dataframe
            try:
                matches = regex.finditer(document[doc])
                test_df = pd.DataFrame([(x.group(), x.start(), x.end()) for x in matches])
                test_df.columns = ['item', 'start', 'end']
                test_df['item'] = test_df.item.str.lower()

                # Get rid of unnesesary charcters from the dataframe
                test_df.replace('&#160;',' ',regex=True,inplace=True)
                test_df.replace('&nbsp;',' ',regex=True,inplace=True)
                test_df.replace(' ','',regex=True,inplace=True)
                test_df.replace('\.','',regex=True,inplace=True)
                test_df.replace('>','',regex=True,inplace=True)
                test_df.replace('<','',regex=True,inplace=True)
                test_df.replace('\n','',regex=True,inplace=True)
                # shift dataset to find the next, next next item
                test_df['next_1'] = test_df['item'].shift(-1,fill_value=0)
                test_df['next_1_start'] = test_df['start'].shift(-1,fill_value=0)
                test_df['next_1_start'] = test_df.next_1_start.astype('int32')
                # test_df['next_1_end'] = test_df['end'].shift(-1,fill_value=0)
                # test_df['next_1_end'] = test_df.next_1_end.astype('int32')
                test_df['pre_1'] = test_df['item'].shift(1,fill_value=0)
                test_df['pre_1_start'] = test_df['start'].shift(1,fill_value=0)
                test_df['pre_1_start'] = test_df.pre_1_start.astype('int32')    
            
            except:
                error = error + 1
                item_1_text = ''
                error_file_list.append(filename[11:13])
                extract_failed_flag = 1

            if not extract_failed_flag:
         
                try:
                    # doc contains 1a 1b section (newr files)
                    if len(test_df[test_df['item'].str.contains('item1a')]):
                        df = test_df[((test_df['item'] == 'item1') & (test_df['next_1'] == 'item1a'))|((test_df['item'] == 'item2') & (test_df['pre_1'] == 'item1b')) | ((test_df['item'] == 'item1') & (test_df['next_1'] == 'item2')) | ((test_df['item'] == 'item2') & (test_df['pre_1'] == 'item1a'))] 
                        df = df[['item', 'start', 'end']]
                        df['next_item'] = df['item'].shift(-1,fill_value=0)
                        df['next_item_start'] = df['start'].shift(-1,fill_value=0)
                        df['next_item_start'] = df.next_item_start.astype('int32')
                        df = df[(df['item']=='item1') & (df['next_item']=='item2')]
                        df['difference'] = df['next_item_start'] - df['start']
                        df.sort_values('difference', ascending = False, inplace = True)
                        item_1_raw = document[doc][df.iloc[0]['start']:df.iloc[0]['next_item_start']]

                    # doc no 1a 1b
                    else:
                        df_1 = test_df.sort_values('start', ascending=True).drop_duplicates(subset=['item'], keep='first')
                        df_1.set_index('item', inplace=True)
                        raw_1 = document[doc][df_1['start'].loc['item1']:df_1['start'].loc['item2']]
                        
                        df_2 = test_df[((test_df['item'] == 'item1') & (test_df['next_1'] == 'item2'))|((test_df['item'] == 'item2') & (test_df['pre_1'] == 'item1'))]
                        df_2 = df_2[['item', 'start', 'end']]
                        df_2['next_item'] = df_2['item'].shift(-1,fill_value=0)
                        df_2['next_item_start'] = df_2['start'].shift(-1,fill_value=0)
                        df_2['next_item_start'] = df_2.next_item_start.astype('int32')
                        df_2 = df_2[(df_2['item']=='item1') & (df_2['next_item']=='item2')]
                        df_2['difference'] = df_2['next_item_start'] - df_2['start']
                        df_2 = df_2[(df_2['item']=='item1')& (df_2['start'] < 100000)]
                        df_2.sort_values('difference', ascending = False, inplace = True)
                        raw_2 = document[doc][df_2.iloc[0]['start']:df_2.iloc[0]['next_item_start']]

                        if (len(raw_1)>=len(raw_2)):
                            item_1_raw = raw_1
                        else:
                            item_1_raw = raw_2

                    item_1_content = BeautifulSoup(item_1_raw, 'lxml')
                    item_1_text = item_1_content.get_text()
                    print(item_1_text[0:1000])

                except:
                    print ("OPOOS!...Something Wrong!")
                    error = error + 1
                    item_1_text = ''
                    error_file_list.append(filename[11:13])


                # path_for_item1_folder = "/home/hongzhuoqiao/10K_Projects/sec_filings/item1_section/" + ticker
            new_file =open(path_for_item1_folder + "/" + file_name+".txt","a")
            new_file.write(item_1_text)
        
            print(">>> New item1 new file for year {} is saved\n".format(file_name))
            count = count + 1

        try:
            shutil.rmtree("/home/hongzhuoqiao/10K_Projects/sec_filings/sec_edgar_filings/"+ticker)
            print ("{}'s folder has been deleted!".format(ticker))
        except OSError as e:
            print("No folder to be deleted or errors in deleting folder.")

    else:
        print ('>>> No 10k files for this company!')

    return file_number, count-1, error,  error_file_list





