import kagglehub

# Download latest version
path = kagglehub.dataset_download("cl0ud0/news-political-bias-classification-dataset")

print("Path to dataset files:", path)