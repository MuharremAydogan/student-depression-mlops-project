import yaml
from pathlib import Path
def readYaml(file_path:Path):
    try:
        with open(file=file_path,mode="r") as f:
            content=yaml.safe_load(f)    

        return content
    except Exception as e:
        raise e    