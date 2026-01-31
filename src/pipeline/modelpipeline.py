from src.entity.config_entity import ModelTrainConfig,ModelEvualtionConfig
from src.compenents.model_trainer import ModelTrain
from src.compenents.model_eval import ModelEvaluation
from pathlib import Path
from src.utils.read_yml import readYaml

def run_model(data_transformation_artf):
    config=readYaml("src/config/config.yml")
    schema=readYaml(Path(config["data"]["validation"]["schema_path"]))
    model_train_conf=ModelTrainConfig(Path(config["data"]["transformation"]["train_transformed_path"]),Path(config["data"]["transformation"]["test_transformed_path"]),model_output_path=Path(config["model"]["model_pickle_path"]))
    model_eval_conf=ModelEvualtionConfig(Path(config["data"]["transformation"]["test_transformed_path"]),target_col=schema["Target"]["Name"],report_path=Path(config["model"]["model_report_path"]))
    print("multi model Train started...")
    model_trainer=ModelTrain(model_train_config=model_train_conf,data_transformation_artifacts=data_transformation_artf,target_col=schema["Target"]["Name"])
    model_train_artf=model_trainer.train()
    print("model train complited...")
    model_eval=ModelEvaluation(model_train_artifacts=model_train_artf,model_eval_config=model_eval_conf)
    print("model Eval Started...")
    model_eval_artf=model_eval.model_evaluation()
    print("model eval complited")

    return model_eval_artf