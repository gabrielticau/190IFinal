import kagglehub

# Download latest version
path = kagglehub.dataset_download("kengoichiki/the-metropolitan-museum-of-art-ukiyoe-dataset")

print("Path to dataset files:", path)
