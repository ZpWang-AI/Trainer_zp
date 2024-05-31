import pandas as pd
import numpy as np
import json
import os
import re
import transformers

from typing import *
from IDRR_data.dataframes import DataFrames
from IDRR_data.ans_word_map import ANS_WORD_LIST, ANS_LABEL_LIST, SUBTYPE_LABEL2ANS_WORD


def ans_words2token(ans_words, tokenizer:transformers.PreTrainedTokenizer):
    vocab = tokenizer.get_vocab()
    if isinstance(tokenizer, transformers.RobertaTokenizer) or \
        isinstance(tokenizer, transformers.RobertaTokenizerFast):
        def ans_word_tokenizer(word):
            return vocab['Ġ'+word.strip()]
    elif isinstance(tokenizer, transformers.BertTokenizer) or \
        isinstance(tokenizer, transformers.BertTokenizerFast):
        def ans_word_tokenizer(word):
            return vocab[word.strip()]
    else:
        raise Exception('wrong type of tokenizer')
    
    return list(map(ans_word_tokenizer, ans_words))

    
class DataFrames2(DataFrames):
    @classmethod
    def from_DataFrames(cls, dataframes:DataFrames):
        return cls(
            data_name=dataframes.data_name,
            label_level=dataframes.label_level,
            relation=dataframes.relation,
            data_path=dataframes.data_path,
        )
    
    def get_dataframe(self, split) -> pd.DataFrame:
        df = self.df[self.df['split']==split]
        if self.relation != 'All':
            df = df[df['relation']==self.relation]
        if self.data_name and self.label_level != 'raw':
            df = self.process_df_conn(df)  # New Feature
            df = self.process_df_sense(df)
        df = df[pd.notna(df['conn1sense1'])]
        df.reset_index(inplace=True)
        return df
    
    def get_ans_word_list(self, tokenizer:transformers.PreTrainedTokenizer=None) -> list:
        ans_word_list = ANS_WORD_LIST[self.data_name] 
        if tokenizer:
            ans_word_list = ans_words2token(ans_words=ans_word_list, tokenizer=tokenizer)
        return ans_word_list
    
    def get_ans_label_list(self, use_label_id=False) -> list:
        ans_label_list = ANS_LABEL_LIST[self.data_name][self.label_level]
        if use_label_id:
            ans_label_list = [self.label_to_id(p)for p in ans_label_list]
        return ans_label_list
        
    def get_subtype_label2ans_word(self) -> dict:
        return SUBTYPE_LABEL2ANS_WORD[self.data_name]
    
    def process_conn(
        self, conn:str, sense:str,
        ans_word_list:list=None, 
        subtype_label2ans_word:dict=None,
        irrelevent_conn=pd.NA,
    ) -> Tuple[str, int]:
        if pd.isna(conn) or pd.isna(sense):
            return irrelevent_conn, irrelevent_conn

        if not ans_word_list:
            ans_word_list = self.get_ans_word_list()
        if not subtype_label2ans_word:
            subtype_label2ans_word = self.get_subtype_label2ans_word()
        
        if conn in ans_word_list:
            return conn, ans_word_list.index(conn)

        sense2 = self.process_sense(
            sense=sense, label_list=list(subtype_label2ans_word.keys()), 
            irrelevent_sense=pd.NA,
        )[0]
        if not pd.isna(sense2):
            conn = subtype_label2ans_word[sense2]
            return conn, ans_word_list.index(conn)
        else:
            raise Exception(f'sense <{sense}> not in subtype_label')
            return irrelevent_conn, irrelevent_conn
    
    def process_df_conn(self, df:pd.DataFrame):
        ans_word_list = self.get_ans_word_list()
        subtype_label2ans_word = self.get_subtype_label2ans_word()
        
        for conn_key in 'conn1 conn2'.split():
            ans_word_values, awid_values = [], []
            for conn, sense in zip(df[conn_key], df[conn_key+'sense1']):
                ans_word, awid = self.process_conn(
                    conn=conn, sense=sense, ans_word_list=ans_word_list,
                    subtype_label2ans_word=subtype_label2ans_word,
                    irrelevent_conn=pd.NA,
                )
                ans_word_values.append(ans_word)
                awid_values.append(awid)
            df[conn_key] = ans_word_values
            df[conn_key+'id'] = awid_values
        return df
    
    
if __name__ == '__main__':
    from pathlib import Path as path
    data_root_path = path('/public/home/hongy/zpwang/IDRR_ConnT5/data')
    pdtb2_df = DataFrames(
        data_name='pdtb2',
        label_level='level2',
        relation='Implicit',
        data_path=data_root_path/'used'/'pdtb2.p1.csv',
    )
    # pdtb3_df = DataFrames(
    #     data_name='pdtb3',
    #     label_level='level2',
    #     relation='Implicit',
    #     data_path=data_root_path/'used'/'pdtb3.p1.csv',
    # )
    # conll_df = DataFrames(
    #     data_name='conll',
    #     label_level='level2',
    #     relation='Implicit',
    #     data_path=data_root_path/'used'/'conll.p1.csv',
    # )

    # sample_df = conll_df
    # sample_df.label_level = 'level2'
    # sample_train_df = sample_df.train_df
    # print(sample_train_df.conn1sense2)
    
    # print(sample_train_df.index)
    # for p in sorted(set(sample_train_df.conn1sense1)):
    #     print(p)
    # if sample_df.label_level != 'raw':
    #     print(len(set(sample_train_df.conn1sense1))==len(sample_df.get_label_list()))
    # print(set(sample_train_df.relation))
    