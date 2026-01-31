from pathlib import Path
from dataclasses import dataclass
from src.entity import artifacts_entity


class data_ingestion_config:
    def __init__(self,data_path:Path,split_ratio:float,train_data_path:Path,test_data_path:Path):
        
        self.data_path=data_path
        self.split_ratio=split_ratio
        self.train_data_path=train_data_path
        self.test_data_path=test_data_path

class data_validation_config:
    def __init__(self,report_path:Path):
        self.report_path=report_path
        


class data_transformation_config:
    def __init__(self,train_transformed_path:Path,test_transformed_path:Path,pickle_obj_path:Path):
        self.train_transformed_path=train_transformed_path
        self.test_transformed_path=test_transformed_path
        self.pickle_obj_path=pickle_obj_path

class ModelTrainConfig:
    def __init__(self,train_df:Path,test_df:Path,model_output_path:Path):
        self.model_output_path=model_output_path
        self.train_df=train_df
        self.test_df=test_df

class ModelEvualtionConfig:
    def __init__(self,test_df_path:Path,target_col:str,report_path:Path):
        self.test_df_path=test_df_path
        self.target_col=target_col
        self.report_path=report_path
        