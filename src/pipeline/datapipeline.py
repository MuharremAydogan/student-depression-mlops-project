from src.compenents.data_ingestion import dataingestion
from src.compenents.data_transformation import DataTransformation
from src.compenents.data_validation import DataValidation
from src.compenents.model_trainer import ModelTrain
from src.compenents.model_eval import ModelEvaluation
from src.entity.config_entity import (data_ingestion_config,data_transformation_config,data_validation_config,ModelEvualtionConfig,ModelTrainConfig)
from src.utils.read_yml import readYaml
from pathlib import Path
def run_data_pipeline():
    config=readYaml("src/config/config.yml")
    schema=readYaml("src/config/schema.yml")
    data_ingestion_conf=data_ingestion_config(Path(config["data"]["ingestion"]["data_path"]),0.2,Path(config["data"]["ingestion"]["train_data_path"]),Path(config["data"]["ingestion"]["test_data_path"]))
    data_transformation_conf=data_transformation_config(Path(config["data"]["transformation"]["train_transformed_path"]),Path(config["data"]["transformation"]["test_transformed_path"]),Path(config["data"]["transformation"]["pickle_obj_path"]))
    data_validation_conf=data_validation_config(Path(config["data"]["validation"]["report_path"]))

    data_ingestion=dataingestion(ingestion_config=data_ingestion_conf)
    print("data ingestion started")
    data_ingestion_artf=data_ingestion.initiate_data_ingestion()
    print("data ingestion complited")
    data_validation=DataValidation(data_ingestion_artifacts=data_ingestion_artf,data_validation_config=data_validation_conf,schema_path=Path(config["data"]["validation"]["schema_path"]))
    print("data validation started...")
    data_validation_artf=data_validation.initiate_data_validation()
    print("data validation complited")


    data_transformer=DataTransformation(data_ingestion_artifacts=data_ingestion_artf,data_transformation_config=data_transformation_conf,target_col=schema["Target"]["Name"])
    print("data transformation started...")
    data_transform_art=data_transformer.transform_data()
    print("data transformation complited...")

    return data_transform_art