import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import re

df = pd.read_csv('../data/train_extra_200.csv', encoding = "ISO-8859-1")
df['length_excerpt'] = df['excerpt'].str.len()
df['num_sentences'] = df['excerpt'].apply(lambda x: len(sent_tokenize(x)))
df['excerpt'] = df['excerpt'].apply(lambda x: re.sub(r'\n+', ' ', x))
df['full_text'] = df['full_text'].apply(lambda x: re.sub(r'\n+', ' ', str(x)))

df['excerpt'] = df['excerpt'].apply(lambda x: re.sub(r'\"\"', '\"', x))
df['full_text'] = df['full_text'].apply(lambda x: re.sub(r'\"\"', '\"', str(x)))

df['len_full_text'] = df['full_text'].str.len()
# print(df.head(100))
# print(df.max())
# print(df.min())

def chunk_indices(sentence_lengths, min_chars=800):
    count = 0
    start_index = 0
    index_bin = []
    for i, v in enumerate(sentence_lengths):
        if count < min_chars:
            count += v
        else:
            index_bin.append((start_index, i))
            start_index = i
            count = v
    return index_bin

def split_text_into_sections(text, target, std_error, min_chars=800):
    if text != 'nan':
        rev1_text = re.sub(r'\[.+?]', '', text)
        cleaned_text = rev1_text
        sentences = sent_tokenize(cleaned_text)
        sentence_lengths = [len(sent) for sent in sentences]
        chunks_idx = chunk_indices(sentence_lengths, min_chars=min_chars)
        chunk_bin = []
        for idx in chunks_idx:
            start = idx[0]
            stop = idx[1]
            paragraph = ''
            for c in range(start, stop):
                paragraph += f'{sentences[c]} '
            chunk_bin.append(paragraph)
        targets = np.full(len(chunk_bin), target)
        std_errors = np.full(len(chunk_bin), std_error)
        df = pd.DataFrame({'excerpt': chunk_bin, 'target': targets, 'standard_error': std_errors})
        return df


extra_texts = [split_text_into_sections(row['full_text'], row['target'], row['standard_error']) for i, row in df.iterrows()]
extra_text_df = pd.concat(extra_texts)
print(extra_text_df)

print(df)

df_combined = pd.concat([df, extra_text_df])
df_combined['target'] = df['target']