from pathlib import Path
from dataclasses import dataclass

@dataclass
class data_ingestion_artifacts:
    train_data_path:Path
    test_data_path:Path

@dataclass
class data_validation_artifacts:
    report_path:Path

@dataclass
class data_transformation_artifacts:
    train_transformed_path:Path
    test_transformed_path:Path
    pickle_obj_path:Path    

@dataclass
class ModelTrainArtifacts:
    model_output_path:Path
    acc:float
    best_model_name:str

@dataclass 
class ModelEvaluationArtifacs:
    confusion_matix:str
    accuracy:float        