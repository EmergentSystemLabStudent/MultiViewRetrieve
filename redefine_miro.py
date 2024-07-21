import sys 
import glob
import shutil
from pathlib import Path
from natsort import natsorted

def main():
    
    dataset_path = "./dataset/MIRO/*"
    object_folders = glob.glob(dataset_path)
    
    for object_folder in object_folders:
        object_name = object_folder.split("/")[-1]
        object_images = natsorted(glob.glob(object_folder + "/*.png"))
        for i in range(0, len(object_images), 160):
            instance_images = object_images[i:i+160]
            instance_id = instance_images[0].split("_")[-2]
            if int(instance_id) <= 5:
                save_folder = Path(f"./dataset/MIRO/train/{object_name}_{instance_id}")
            else:
                save_folder = Path(f"./dataset/MIRO/eval/{object_name}_{instance_id}")
                
            if not save_folder.exists():
                save_folder.mkdir(parents=True)
            for instance_image in instance_images:
                shutil.move(instance_image, str(save_folder / (Path(instance_image).stem + ".png")))
        Path(object_folder).resolve().rmdir()
        
        
if __name__ == "__main__":
    main()