import sys 
import shutil 
from pathlib import Path

from tqdm import tqdm

def main():
    
    dataset_folder = Path("./dataset/ObjectPI/eval")
    for class_folder in dataset_folder.iterdir():
        for instance_folder in class_folder.iterdir():
            shutil.move(str(instance_folder), "./dataset/ObjectPI/eval")
        class_folder.rmdir()
    
    
if __name__ == "__main__":
    main()