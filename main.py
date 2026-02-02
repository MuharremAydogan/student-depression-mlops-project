from src.pipeline.datapipeline import run_data_pipeline
from src.pipeline.modelpipeline import run_model

def main():
    transform_artf=run_data_pipeline()
    eval_artf=run_model(transform_artf)

