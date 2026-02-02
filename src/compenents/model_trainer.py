from src.entity.config_entity import ModelTrainConfig
from src.entity.artifacts_entity import ModelTrainArtifacts,data_transformation_artifacts

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
import pickle
import json
from src.cloud.s3_syncer import S3Sync
class ModelTrain:
    def __init__(self,model_train_config:ModelTrainConfig,data_transformation_artifacts:data_transformation_artifacts,target_col:str):
        self.model_train_config=model_train_config
        self.data_transformation_artifacts=data_transformation_artifacts
        self.target_col=target_col
        self.s3_sync=S3Sync()
    def models(self):
        models={"rf":RandomForestClassifier(random_state=42),"adaboost":AdaBoostClassifier(random_state=42),"gb":GradientBoostingClassifier(random_state=42)}
        model_params = {

        "rf": {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        },

        "adaboost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0]
        },

        "gb": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
            "min_samples_split": [2, 5]
        }

    }   
        return models,model_params
    
    def get_data(self):
        train_df=pd.read_csv(Path(self.data_transformation_artifacts.train_transformed_path))
        test_df=pd.read_csv(Path(self.data_transformation_artifacts.test_transformed_path))
        x_train=train_df.drop([self.target_col],axis=1)
        y_train=train_df[self.target_col]
        x_test=test_df.drop([self.target_col],axis=1)
        y_test=test_df[self.target_col]

        return x_train,y_train,x_test,y_test
        
    
    def train(self):

        models, model_params = self.models()
        x_train, y_train, x_test, y_test = self.get_data()

        best_acc = 0
        best_model = None
        best_model_name = None
        best_params=None
        for name, model in models.items():

            params = model_params[name]

            grid_model = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=3,
                n_jobs=-1,
                scoring="accuracy"
            )

            grid_model.fit(x_train, y_train)

            y_pred = grid_model.best_estimator_.predict(x_test)
            acc = accuracy_score(y_test, y_pred)

            if acc > best_acc:
                best_acc = acc
                best_model = grid_model.best_estimator_
                best_model_name = name
                best_params=grid_model.best_params_

        print("Best model:", best_model_name)
        print("Best accuracy:", best_acc)

        model_param_path="artifacts/objects/model_param.json"
        model_param_path=Path(model_param_path)
        model_param_path.parent.mkdir(parents=True,exist_ok=True)
        
        with open(model_param_path,mode="w") as f:
            json.dump(best_params,f,indent=4)


        model_output_path=self.model_train_config.model_output_path
        model_output_path.parent.mkdir(parents=True,exist_ok=True)

        with open(model_output_path,mode="wb") as f:
            pickle.dump(best_model,f)

        model_train_artifacts=ModelTrainArtifacts(model_output_path=model_output_path,acc=best_acc,best_model_name=best_model_name)

        #model push to s3
        folder="artifacts"
        try:
            aws_bucket_url=f"s3://studentdepression"
            self.s3_sync.sync_folder_to_s3(folder=folder,aws_bucket_url=aws_bucket_url)
        except Exception as e :
            raise e    

        return model_train_artifacts

        





        

    