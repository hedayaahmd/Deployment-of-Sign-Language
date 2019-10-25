
TRAINING_DATA_URL="hedayaahmd/sign-language-word-dataset"
NOW=$(date)

kaggle datasets download -d $TRAINING_DATA_URL -p packages/CNNModel/CNNModel/datasets/ && \
echo "downloaded in : packages/CNNModel/CNNModel/datasets " > packages/CNNModel/CNNModel/datasets/training_data_reference.txt
unzip packages/CNNModel/CNNModel/datasets/sign-language-word-dataset.zip -d packages/CNNModel/CNNModel/datasets/sign-language-word-dataset && \
echo $TRAINING_DATA_URL 'retrieved on:' $NOW > packages/CNNModel/CNNModel/datasets/training_data_reference.txt
