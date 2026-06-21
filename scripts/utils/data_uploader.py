import kagglehub, os

def load_dataset_from_src(dataset:str, dir:str):
    if os.path.exists(dir) and os.listdir(dir):
        print("Dataset currently exists!")
    else :
        path = kagglehub.dataset_download(dataset, output_dir=dir)
        print("Path to dataset files:", path)

if __name__ == "__main__":
    load_dataset_from_src("balraj98/modelnet10-princeton-3d-object-dataset", "data")
