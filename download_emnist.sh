TARGET_DIR="data"
mkdir -p "$TARGET_DIR"
curl -L -o emnist.zip https://www.kaggle.com/api/v1/datasets/download/crawford/emnist
unzip -q emnist.zip -d "$TARGET_DIR"
rm emnist.zip