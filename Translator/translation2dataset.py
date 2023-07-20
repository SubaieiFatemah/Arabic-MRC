'''#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gdown
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import sys

# In[2]:

#from Preprocess.arabertpreprocess import ArabertPreprocessor

from arabert.preprocess import ArabertPreprocessor


# In[3]:

'''
'''
url = 'https://drive.google.com/uc?id=1-3aZrNqg_TwX2WbPP7-BmDR1HwCMsdiL'
output = 'Asquadv2-train.csv'
gdown.download(url, output, quiet=False)
'''
'''

# In[3]:

file_name = sys.argv[1]
df = pd.read_csv(file_name)


# In[4]:


print(df.shape)
print(df.columns)


# In[5]:


filt = df['answer_start']!=-1
cleaned_span_df = df[filt]
print(cleaned_span_df.shape)


# In[9]:


cleaned_span_df = cleaned_span_df.dropna(subset = ['title', 'context'])


# In[10]:


def dataframe2dict(df):
    model_name = "araelectra-base-discriminator"
    arabert_prep = ArabertPreprocessor(model_name=model_name)
    generated_data = dict()
    generated_data['version'] =2.0
    generated_data['data'] = list()
    df_title_g = df.groupby(['title'])
    title_keys = list(df_title_g.groups.keys())
    cnt = 0
    for key1 in title_keys:# first level
        new_df = df_title_g.get_group(key1)
        df_context_g = new_df.groupby(by=['context'])
        context_keys = list(df_context_g.groups.keys())
        key1_processed = arabert_prep.preprocess(key1)
        generated_data['data'].append({'title':key1_processed,'paragraphs':list()})
        for key2 in context_keys:#second level
            new_df_2 = df_context_g.get_group(key2)
            key2_processed = arabert_prep.preprocess(key2)
            triplet_dict = {'context':key2_processed, 'qas':list()}
            for idx, row in new_df_2.iterrows():
                qa_dict = {'question':arabert_prep.preprocess(row['question']), 'id':str(cnt), 'is_impossible':row['is_impossible']}
                if row['is_impossible'] ==False:
                    qa_dict['answers'] = [{'text':arabert_prep.preprocess(row['answer']), 'answer_start':row['answer_start']}]
                else:
                    qa_dict['plausible_answers'] =[{'text':arabert_prep.preprocess(row['answer']), 'answer_start':0}]
                    qa_dict['answers']=list()
                triplet_dict['qas'].append(qa_dict)
                cnt = cnt+1
            generated_data['data'][-1]['paragraphs'].append(triplet_dict)
    return generated_data


# In[11]:


df_train, df_test, y_train, y_test = train_test_split(cleaned_span_df, cleaned_span_df['is_impossible'],test_size = 0.2,stratify = cleaned_span_df['is_impossible'])


# In[12]:


# 0.111111 x 0.9 = 0.1
df_train, df_val, y_train, y_val = train_test_split(df_train, df_train['is_impossible'],test_size = 0.25,stratify = df_train['is_impossible'])


# In[13]:


print(df_train.shape, df_val.shape, df_test.shape)


# In[18]:


train_dataset = dataframe2dict(df_train)
val_dataset = dataframe2dict(df_val)
test_dataset = dataframe2dict(df_test)


# In[19]:


with open("/content/asquadv2-train.json", "w") as outfile:
    json.dump(train_dataset, outfile)
with open("/content/asquadv2-val.json", "w") as outfile:
    json.dump(val_dataset, outfile)
with open("/content/asquadv2-test.json", "w") as outfile:
    json.dump(test_dataset, outfile)
'''
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import sys
from arabert.preprocess import ArabertPreprocessor

file_name = sys.argv[1]
df = pd.read_csv(file_name)

print(df.shape)
print(df.columns)

def dataframe2dict(df):
    model_name = "araelectra-base-discriminator"
    arabert_prep = ArabertPreprocessor(model_name=model_name)
    generated_data = dict()
    generated_data['version'] = 2.0
    generated_data['data'] = []
    cnt = 0
    for _, row in df.iterrows():
        context = arabert_prep.preprocess(row['context'])
        question = arabert_prep.preprocess(row['question'])
        answer = arabert_prep.preprocess(row['answer'])
        qa_dict = {
            'question': question,
            'id': str(cnt),
            'is_impossible': row['is_impossible']
        }
        if not row['is_impossible']:
            qa_dict['answers'] = [{'text': answer, 'answer_start': row['answer_start']}]
        else:
            qa_dict['answers'] = []
        paragraph = {
            'context': context,
            'qas': [qa_dict]
        }
        generated_data['data'].append({'title': '', 'paragraphs': [paragraph]})
        cnt += 1
    return generated_data

random_state=random_state = 42
df_train, df_test, y_train, y_test = train_test_split(df, df['is_impossible'], test_size=0.3, random_state=random_state)
df_val, df_test, y_val, y_test = train_test_split(df_test, df_test['is_impossible'], test_size=0.5, random_state=random_state)


print(df_train.shape, df_val.shape, df_test.shape)

train_dataset = dataframe2dict(df_train)
val_dataset = dataframe2dict(df_val)
test_dataset = dataframe2dict(df_test)

with open("cgsqa-train.json", "w") as outfile:
    json.dump(train_dataset, outfile)
with open("cgsqa-val.json", "w") as outfile:
    json.dump(val_dataset, outfile)
with open("cgsqa-test.json", "w") as outfile:
    json.dump(test_dataset, outfile)
