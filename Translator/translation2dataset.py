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
