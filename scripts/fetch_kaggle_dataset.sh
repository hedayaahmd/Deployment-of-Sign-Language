
TRAINING_DATA_URL="hedayaahmd/sign-language-word-dataset"
NOW=$(date)

kaggle datasets download -d $TRAINING_DATA_URL -p packages/CNNModel/CNNModel/datasets/ && \
unzip packages/CNNModel/CNNModel/datasets/sign-language-word-dataset.zip -d packages/CNNModel/CNNModel/datasets/sign-language-word-dataset && \
echo $TRAINING_DATA_URL 'retrieved on:' $NOW > packages/CNNModel/CNNModel/datasets/training_data_reference.txt && \
mkdir -p "./packages/CNNModel/CNNModel/datasets/sign-language-word-dataset/angry"  && \
mv -v "./packages/CNNModel/CNNModel/datasets/sign-language-word-dataset/buy/"* "./packages/CNNModel/CNNModel/datasets/sign-language-word-dataset/how you"
rm -rf "./packages/CNNModel/CNNModel/datasets/sign-language-word-dataset/thanks"
