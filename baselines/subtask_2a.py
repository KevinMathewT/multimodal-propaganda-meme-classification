import pandas as pd
import random
import logging
import argparse
from os.path import join, dirname, basename
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import csv
import json

import sys
sys.path.append('.')

from scorer.task2 import evaluate
from format_checker.task2 import check_format

random.seed(0)
ROOT_DIR = dirname(dirname(__file__))

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

def read_data(data_fpath):
    data = {'id': [], 'img_path': [], 'text': [], 'class_label': []}
    with open(data_fpath, encoding='utf-8') as f:
        js_obj = json.load(f)
        for entry in js_obj:
            data['id'].append(entry['id'])
            data['img_path'].append(entry['img_path'])
            data['text'].append(entry['text'])
            data['class_label'].append(entry['class_label'])
    return data

def run_majority_baseline(data_fpath, test_fpath, results_fpath):
    # train_df = pd.read_csv(data_fpath, dtype=object, encoding="utf-8", sep='\t')
    # test_df = pd.read_csv(test_fpath, dtype=object, encoding="utf-8", sep='\t')
    train_df = read_data(data_fpath)
    test_df = read_data(test_fpath)

    pipeline = DummyClassifier(strategy="most_frequent")

    text_head = "text"
    id_head = "id"

    pipeline.fit(train_df[text_head], train_df['class_label'])

    with open(results_fpath, "w") as results_file:
        predicted_distance = pipeline.predict(test_df[text_head])

        results_file.write("id\tclass_label\trun_id\n")

        for i, line in enumerate(test_df[id_head]):
            label = predicted_distance[i]

            results_file.write("{}\t{}\t{}\n".format(line, label, "majority"))


def run_random_baseline(data_fpath, results_fpath):
    # gold_df = pd.read_csv(data_fpath,  dtype=object, encoding="utf-8", sep='\t')
    gold_df = read_data(data_fpath)
    #label_list=gold_df['class_label'].to_list()
    label_list= ["propaganda", "not_propaganda"]

    id_head = "id"

    with open(results_fpath, "w") as results_file:
        results_file.write("id\tclass_label\trun_id\n")
        for i, line in enumerate(gold_df[id_head]):
            results_file.write('{}\t{}\t{}\n'.format(line,random.choice(label_list), "random"))


def run_ngram_baseline(train_fpath, test_fpath, results_fpath):
    # train_df = pd.read_csv(train_fpath, dtype=object, encoding="utf-8", sep='\t')
    # test_df = pd.read_csv(test_fpath, dtype=object, encoding="utf-8", sep='\t')
    train_df = read_data(train_fpath)
    test_df = read_data(test_fpath)

    text_head = "text"
    id_head = "id"


    pipeline = Pipeline([
        ('ngrams', TfidfVectorizer(ngram_range=(1, 1),lowercase=True,use_idf=True,max_df=0.95, min_df=3,max_features=5000)),
        ('clf', SVC(C=1, gamma='scale', kernel='linear', random_state=0))
    ])
    pipeline.fit(train_df[text_head], train_df['class_label'])

    with open(results_fpath, "w") as results_file:
        predicted_distance = pipeline.predict(test_df[text_head])
        results_file.write("id\tclass_label\trun_id\n")
        for i, line in enumerate(test_df[id_head]):
            label = predicted_distance[i]
            results_file.write("{}\t{}\t{}\n".format(line, label, "ngram"))


def run_baselines(train_fpath, test_fpath):
    majority_baseline_fpath = join(ROOT_DIR,
                                 f'data/majority_baseline_{basename(test_fpath).replace("json", "tsv")}')
    run_majority_baseline(train_fpath, test_fpath, majority_baseline_fpath)

    if check_format(majority_baseline_fpath):
        acc, precision, recall, f1 = evaluate(test_fpath, majority_baseline_fpath)
        logging.info(f"Majority Baseline F1 (macro): {f1}")


    random_baseline_fpath = join(ROOT_DIR, f'data/random_baseline_{basename(test_fpath).replace("json", "tsv")}')
    run_random_baseline(test_fpath, random_baseline_fpath)

    if check_format(random_baseline_fpath):
        acc, precision, recall, f1 = evaluate(test_fpath, random_baseline_fpath)
        logging.info(f"Random Baseline F1 (macro): {f1}")

    ngram_baseline_fpath = join(ROOT_DIR, f'data/ngram_baseline_{basename(test_fpath).replace("json", "tsv")}')
    run_ngram_baseline(train_fpath, test_fpath, ngram_baseline_fpath)
    if check_format(ngram_baseline_fpath):
        acc, precision, recall, f1 = evaluate(test_fpath, ngram_baseline_fpath)
        logging.info(f"Ngram Baseline F1 (macro): {f1}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file-path", "-t", required=True, type=str,
                        help="The absolute path to the training data")
    parser.add_argument("--dev-file-path", "-d", required=True, type=str,
                        help="The absolute path to the dev data")

    args = parser.parse_args()
    run_baselines(args.train_file_path, args.dev_file_path)
