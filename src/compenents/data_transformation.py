from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pickle
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.entity.artifacts_entity import data_transformation_artifacts,data_ingestion_artifacts,data_validation_artifacts
from src.entity.config_entity import data_transformation_config
import pandas as pd
class DataTransformation:
    def __init__(self,data_ingestion_artifacts:data_ingestion_artifacts,data_transformation_config:data_transformation_config,target_col:str):
        self.data_ingestion_artifacts=data_ingestion_artifacts
        self.data_transformation_config=data_transformation_config
        self.target_col=target_col

    def transform_data(self):
        train_path = self.data_ingestion_artifacts.train_data_path
        test_path = self.data_ingestion_artifacts.test_data_path

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        obj_col = train_df.select_dtypes(include="object").columns.tolist()
        num_col = [col for col in train_df.columns if col not in obj_col and col != self.target_col]

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        processor = ColumnTransformer(
            transformers=[
                ("num_pipeline", num_pipeline, num_col),
                ("cat_pipeline", cat_pipeline, obj_col)
            ]
        )

        X_train = train_df.drop(self.target_col, axis=1)
        y_train = train_df[self.target_col]

        X_test = test_df.drop(self.target_col, axis=1)
        y_test = test_df[self.target_col]

        X_train_arr = processor.fit_transform(X_train)
        X_test_arr = processor.transform(X_test)

        
        try:
            feature_names = processor.get_feature_names_out()
            X_train_df = pd.DataFrame(X_train_arr, columns=feature_names)
            X_test_df = pd.DataFrame(X_test_arr, columns=feature_names)
        except:
            X_train_df = pd.DataFrame(X_train_arr)
            X_test_df = pd.DataFrame(X_test_arr)

        train_transformed_df = pd.concat(
            [X_train_df.reset_index(drop=True),
            y_train.reset_index(drop=True)],
            axis=1
        )

        test_transformed_df = pd.concat(
            [X_test_df.reset_index(drop=True),
            y_test.reset_index(drop=True)],
            axis=1
        )

        train_path = self.data_transformation_config.train_transformed_path
        test_path = self.data_transformation_config.test_transformed_path
        pickle_path = self.data_transformation_config.pickle_obj_path

        train_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        pickle_path.parent.mkdir(parents=True, exist_ok=True)

        train_transformed_df[self.target_col] = train_transformed_df[self.target_col].astype(int)
        test_transformed_df[self.target_col] = test_transformed_df[self.target_col].astype(int)

        train_transformed_df.to_csv(train_path, index=False)
        test_transformed_df.to_csv(test_path, index=False)

        with open(pickle_path, "wb") as f:
            pickle.dump(processor, f)

        transformation_artifacts = data_transformation_artifacts(
            train_transformed_path=train_path,
            test_transformed_path=test_path,
            pickle_obj_path=pickle_path
        )

        return transformation_artifacts
