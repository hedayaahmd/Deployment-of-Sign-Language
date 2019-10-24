
TRAINING_DATA_URL="hedayaahmd/sign-language-word-dataset"
NOW=$(date)

kaggle datasets download -d $TRAINING_DATA_URL -p packages/CNNModel/CNNModel/datasets/ && \
unzip packages/CNNModel/CNNModel/datasets/sign-language-word-dataset.zip -d packages/CNNModel/CNNModel/datasets/sign-language-word-dataset 
