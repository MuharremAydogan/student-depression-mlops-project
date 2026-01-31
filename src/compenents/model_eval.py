from src.entity.artifacts_entity import ModelEvaluationArtifacs,ModelTrainArtifacts
from src.entity.config_entity import ModelEvualtionConfig

from sklearn.metrics import confusion_matrix,accuracy_score
import pickle
from pathlib import Path
import pandas as pd
import mlflow.sklearn
import mlflow
import json
class ModelEvaluation:
    def __init__(self,model_train_artifacts:ModelTrainArtifacts,model_eval_config:ModelEvualtionConfig):
        self.model_train_artifacts=model_train_artifacts
        self.model_eval_config=model_eval_config

    def eval(self,y_test,y_pred):
        acc=accuracy_score(y_test,y_pred)
        conf=str(confusion_matrix(y_test,y_pred))    
        
        return acc,conf
    
    def model_evaluation(self):
        model_path=self.model_train_artifacts.model_output_path
        model_name=self.model_train_artifacts.best_model_name
        
        with open(Path(model_path),mode="rb") as f:
            model=pickle.load(f)

        target_col=self.model_eval_config.target_col
        test_df_path=self.model_eval_config.test_df_path

        test_df=pd.read_csv(Path(test_df_path))
        x_test=test_df.drop([target_col],axis=1)
        y_test=test_df[target_col]

        y_pred=model.predict(x_test)

        acc=accuracy_score(y_test,y_pred)
        cm=str(confusion_matrix(y_test,y_pred))

        report={"acc":acc,"cm":cm}

        reprot_path=self.model_eval_config.report_path
        
        reprot_path.parent.mkdir(parents=True,exist_ok=True)

        with open(reprot_path,mode="w") as f:
            json.dump(report,f,indent=4)

        modelval_artifacts=ModelEvaluationArtifacs(confusion_matix=cm,accuracy=acc) 
        return modelval_artifacts   

