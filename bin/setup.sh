pip install poetry

cd multimodal-propaganda-meme-classification
wget https://gitlab.com/araieval/araieval_arabicnlp24/-/raw/main/task2/data/arabic_memes_propaganda_araieval_24_train.json
wget https://gitlab.com/araieval/araieval_arabicnlp24/-/raw/main/task2/data/arabic_memes_propaganda_araieval_24_dev.json
wget https://gitlab.com/araieval/araieval_arabicnlp24/-/raw/main/task2/data/arabic_memes_araieval_24_train_dev.tar.gz

tar -xvzf arabic_memes_araieval_24_train_dev.tar.gz

poetry install
poetry shell

python example_scripts/Multimodal_example_task2C.py
python scorer/task2.py --gold-file-path data/arabic_memes_propaganda_araieval_24_dev.json --pred-file-path task2C_kevinmathew.tsv 