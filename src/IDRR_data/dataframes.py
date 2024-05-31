import pandas as pd
import numpy as np
import json
import os
import re

from typing import *
from IDRR_data.label_list2 import LEVEL1_LABEL_LIST, LEVEL2_LABEL_LIST


class DataFrames:
    def __init__(
        self,
        data_name:Literal['pdtb2', 'pdtb3', 'conll']=None,
        label_level:Literal['level1', 'level2', 'raw']='raw',
        relation:Literal['Implicit', 'Explicit', 'All']='Implicit',
        data_path:str=None,
        # label_use_id=False
    ) -> None:
        assert data_name in ['pdtb2', 'pdtb3', 'conll']
        assert label_level in ['level1', 'level2', 'raw']
        assert relation in ['Implicit', 'Explicit', 'All']
        self.data_name = data_name
        self.label_level = label_level
        self.relation = relation 
        self.data_path = data_path
        # self.label_use_id = label_use_id
    
        self.df = pd.DataFrame()
        if data_path:
            self.build_dataframe(data_path=data_path)
    
    def build_dataframe(self, data_path):
        self.df = pd.read_csv(data_path, low_memory=False)
    
    def get_dataframe(self, split) -> pd.DataFrame:
        df = self.df[self.df['split']==split]
        if self.relation != 'All':
            df = df[df['relation']==self.relation]
        if self.data_name and self.label_level != 'raw':
            df = self.process_df_sense(df)
        df = df[pd.notna(df['conn1sense1'])]
        df.reset_index(inplace=True)
        return df
    
    @property
    def train_df(self) -> pd.DataFrame:
        return self.get_dataframe('train')
    
    @property
    def dev_df(self) -> pd.DataFrame:
        return self.get_dataframe('dev')
        
    @property
    def test_df(self) -> pd.DataFrame:
        return self.get_dataframe('test')
            
    def get_label_list(self):
        if self.label_level == 'level1':
            label_list = LEVEL1_LABEL_LIST
        elif self.label_level == 'level2':
            label_list = LEVEL2_LABEL_LIST[self.data_name]
        else:
            raise Exception('wrong label_level')
        return label_list     
    
    @property
    def label_list(self) -> List[str]:
        return self.get_label_list()
           
    def label_to_id(self, label):
        label_list = self.get_label_list()
        return label_list.index(label)
    
    def id_to_label(self, lid):
        label_list = self.get_label_list()
        return label_list[lid]
    
    def process_sense(
        self, sense:str,
        label_list=None, 
        irrelevent_sense=pd.NA,
    ) -> Tuple[str, int]:
        """
        match the longest label
        """
        if pd.isna(sense):
            return (irrelevent_sense,)*2 

        if not label_list:
            label_list = self.get_label_list()
        
        res_lid = -1
        max_label_len = -1
        for lid, label in enumerate(label_list):
            if sense.startswith(label):
                if len(label) > max_label_len:
                    res_lid = lid
                    max_label_len = len(label)
        
        if res_lid == -1: 
            return (irrelevent_sense,)*2  
        else:
            return label_list[res_lid], res_lid
        
    def process_df_sense(self, df:pd.DataFrame):
        label_list = self.get_label_list()
        
        for sense_key in 'conn1sense1 conn1sense2 conn2sense1 conn2sense2'.split():
            label_values, lid_values = [], []
            for sense in df[sense_key]:
                label, lid = self.process_sense(
                    sense=sense, label_list=label_list, irrelevent_sense=pd.NA,
                )
                label_values.append(label)
                lid_values.append(lid)
            df[sense_key] = label_values
            df[sense_key+'id'] = lid_values
        return df
    
