pip install poetry
apt-get update
apt-get upgrade
apt-get install zip

cd multimodal-propaganda-meme-classification
wget https://gitlab.com/araieval/araieval_arabicnlp24/-/raw/main/task2/data/arabic_memes_propaganda_araieval_24_train.json
wget https://gitlab.com/araieval/araieval_arabicnlp24/-/raw/main/task2/data/arabic_memes_propaganda_araieval_24_dev.json
wget https://gitlab.com/araieval/araieval_arabicnlp24/-/raw/main/task2/data/arabic_memes_araieval_24_train_dev.tar.gz

tar -xvzf arabic_memes_araieval_24_train_dev.tar.gz

poetry install
poetry shell

git pull
clear
python example_scripts/Multimodal_example_task2C.py
python scorer/task2.py --gold-file-path data/arabic_memes_propaganda_araieval_24_dev.json --pred-file-path task2C_kevinmathew.tsv 
rm -rf task2C_kevinmathew.zip
zip task2C_kevinmathew.zip task2C_kevinmathew.tsv

git pull
clear
python example_scripts/DistilBERT_example_task2A.py
python scorer/task2.py --gold-file-path data/arabic_memes_propaganda_araieval_24_dev.json --pred-file-path task2A_kevinmathew.tsv 
rm -rf task2A_kevinmathew.zip
zip task2A_kevinmathew.zip task2A_kevinmathew.tsv
