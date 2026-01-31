import pandas as pd
from pathlib import Path
import json
from src.entity.config_entity import data_validation_config
from src.entity.artifacts_entity import data_ingestion_artifacts,data_validation_artifacts
from src.utils.read_yml import readYaml
class DataValidation:
    def __init__(self,data_ingestion_artifacts:data_ingestion_artifacts,data_validation_config:data_validation_config,schema_path:Path):
        self.data_ingestion_artifacts=data_ingestion_artifacts
        self.data_validation_config=data_validation_config
        self.schema_path=schema_path

    def read_data(self):
        train_df_path=self.data_ingestion_artifacts.train_data_path
        test_df_path=self.data_ingestion_artifacts.test_data_path
        train_df=pd.read_csv(train_df_path)
        test_df=pd.read_csv(test_df_path)
        return train_df,test_df
    
    def validate_column(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        schema: dict = readYaml(self.schema_path)

        schema_columns = schema["Columns"]

        
        if len(schema_columns) != train_df.shape[1] or len(schema_columns) != test_df.shape[1]:
            return False

        for col_name, col_type in schema_columns.items():

           
            if col_name not in train_df.columns or col_name not in test_df.columns:
                return False

            
            train_dtype = str(train_df[col_name].dtype)
            test_dtype = str(test_df[col_name].dtype)

            if train_dtype != col_type or test_dtype != col_type:
                return False

        return True
    def initiate_data_validation(self):
        train_df, test_df = self.read_data()

        ret = self.validate_column(train_df=train_df, test_df=test_df)

        report_path = self.data_validation_config.report_path

        result = {
            "validated_columns": list(train_df.columns),
            "is_valid": ret
        }

        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(result, f, indent=4)

        data_val_artifacts = data_validation_artifacts(report_path=report_path)
        return data_val_artifacts