from src.utils.read_yml import readYaml
from src.entity.config_entity import (data_ingestion_config)
from src.entity.artifacts_entity import (data_ingestion_artifacts)

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

class dataingestion:
    def __init__(self,ingestion_config:data_ingestion_config):
        self.data_ingestion_config=ingestion_config

    def read_data(self):
        return pd.read_csv(self.data_ingestion_config.data_path)
    
    def split_data(self):
        split_ratio=self.data_ingestion_config.split_ratio
        df=self.read_data()

        train_df,test_df=train_test_split(df,test_size=split_ratio,random_state=42)
        return train_df,test_df
    
    def initiate_data_ingestion(self):
        train_df,test_df=self.split_data()
        train_path=self.data_ingestion_config.train_data_path
        test_path=self.data_ingestion_config.test_data_path

        train_path.parent.mkdir(parents=True,exist_ok=True)
        test_path.parent.mkdir(parents=True,exist_ok=True)
        train_df=train_df.drop(["Student_ID"],axis=1)
        test_df=test_df.drop(["Student_ID"],axis=1)
        train_df.to_csv(train_path,index=False)
        test_df.to_csv(test_path,index=False)

        ingestion_artifacts=data_ingestion_artifacts(train_data_path=train_path,test_data_path=test_path)
        return ingestion_artifacts



    