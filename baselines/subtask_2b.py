import jsonlines
import json
import random
import logging
import argparse
from os.path import join, dirname, basename
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

import sys
sys.path.append('.')

from scorer.task2 import evaluate
from format_checker.task2 import check_format

random.seed(100)
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

def run_majority_baseline(data_dir, train_fpath, test_fpath, results_fpath, split):
    train_objs = read_data(join(data_dir, train_fpath))
    test_objs = read_data(join(data_dir, test_fpath))

    tr_feats = json.load(open(join(data_dir, "features", "train_feats.json")))
    te_feats = json.load(open(join(data_dir, "features", "%s_feats.json"%(split))))

    train_feats = [tr_feats['imgfeats'][ids] for ids in train_objs['id']]
    train_label = train_objs['class_label']

    test_feats = [te_feats['imgfeats'][ids] for ids in test_objs['id']]
    test_label = test_objs['class_label']


    pipeline = DummyClassifier(strategy="most_frequent")
    pipeline.fit(train_feats, train_label)

    with open(results_fpath, "w") as results_file:
        predicted_distance = pipeline.predict(test_feats)
        results_file.write("id\tclass_label\trun_id\n")

        for i, line in enumerate(test_objs['id']):
            label = predicted_distance[i]

            results_file.write("{}\t{}\t{}\n".format(line, label, "majority"))


def run_random_baseline(data_dir, data_fpath, results_fpath):
    gold_objs = read_data(join(data_dir, data_fpath))
    #label_list= [obj["class_label"] for obj in gold_objs]
    label_list= ["propaganda","not_propaganda"]


    with open(results_fpath, "w") as results_file:
        results_file.write("id\tclass_label\trun_id\n")
        for i, line in enumerate(gold_objs['id']):
            results_file.write('{}\t{}\t{}\n'.format(line, random.choice(label_list), "random"))


def run_resnet_baseline(data_dir, split, train_fpath, test_fpath, results_fpath):
    tr_feats = json.load(open(join(data_dir, "features", "train_feats.json")))
    te_feats = json.load(open(join(data_dir, "features", "%s_feats.json"%(split))))
    train_df = read_data(join(data_dir, train_fpath))
    test_df = read_data(join(data_dir, test_fpath))
    # train_id_lab = [[obj["tweet_id"], obj["class_label"]] for obj in jsonlines.open(join(data_dir, train_fpath))]
    # test_id_lab = [[obj["tweet_id"], obj["class_label"]] for obj in jsonlines.open(join(data_dir, test_fpath))]
    train_id_lab = [train_df['id'], train_df['class_label']]
    test_id_lab = [test_df['id'], test_df['class_label']]

    # tr_cat_feats = [tr_feats["imgfeats"][obj[0]]+tr_feats["textfeats"][obj[0]] for obj in train_id_lab]
    # tr_cat_feats = np.array(tr_cat_feats)
    # te_cat_feats = [te_feats["imgfeats"][obj[0]]+te_feats["textfeats"][obj[0]] for obj in test_id_lab]
    # te_cat_feats = np.array(te_cat_feats)
    tr_cat_feats = [tr_feats["imgfeats"][obj] for obj in train_id_lab[0]]
    tr_cat_feats = np.array(tr_cat_feats)
    te_cat_feats = [te_feats["imgfeats"][obj] for obj in test_id_lab[0]]
    te_cat_feats = np.array(te_cat_feats)

    clf = SVC(C=1, kernel='linear', random_state=0)
    clf.fit(tr_cat_feats, train_id_lab[1])

    with open(results_fpath, "w") as results_file:
        predicted_distance = clf.predict(te_cat_feats)
        results_file.write("id\tclass_label\trun_id\n")

        for i, line in enumerate(test_id_lab[0]):
            label = predicted_distance[i]

            results_file.write("{}\t{}\t{}\n".format(line, label, "imgbert"))



def run_baselines(data_dir, test_split, train_fpath, test_fpath):
    ## Write test file in format
    # test_objs = [obj for obj in jsonlines.open(join(data_dir, test_fpath))]
    gold_fpath = join(data_dir, f'{basename(test_fpath)}')

    majority_baseline_fpath = join(data_dir,
                                 f'majority_baseline_{basename(test_fpath.replace("json", "tsv"))}')
    run_majority_baseline(data_dir, train_fpath, test_fpath, majority_baseline_fpath, test_split)
    if check_format(majority_baseline_fpath):
        acc, precision, recall, f1 = evaluate(gold_fpath, majority_baseline_fpath)
        logging.info(f"Majority Baseline F1 (macro): {f1}")

    random_baseline_fpath = join(data_dir, f'random_baseline_{basename(test_fpath.replace("json", "tsv"))}')
    run_random_baseline(data_dir, test_fpath, random_baseline_fpath)
    if check_format(random_baseline_fpath):
        acc, precision, recall, f1 = evaluate(gold_fpath, random_baseline_fpath)
        logging.info(f"Random Baseline for F1 (macro): {f1}")

    resnet_baseline_fpath = join(data_dir, f'imgbert_baseline_{basename(test_fpath.replace("json", "tsv"))}')
    run_resnet_baseline(data_dir, test_split, train_fpath, test_fpath, resnet_baseline_fpath)
    if check_format(resnet_baseline_fpath):
        acc, precision, recall, f1 = evaluate(gold_fpath, resnet_baseline_fpath)
        logging.info(f"ResNet Baseline F1 (macro): {f1}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", "-d", required=True, type=str,
                        default="~/araieval_arabicnlp24/task2/",
                        help="The absolute path to the training data")
    parser.add_argument("--test-split", "-s", required=True, type=str,
                        default="dev", help="Test split name")
    parser.add_argument("--train-file-name", "-tr", required=True, type=str,
                        default="arabic_memes_propaganda_araieval_24_train.json",
                        help="Training file name")
    parser.add_argument("--test-file-name", "-te", required=True, type=str,
                        default="arabic_memes_propaganda_araieval_24_dev.json",
                        help="Test file name")
    args = parser.parse_args()
    run_baselines(args.data_dir, args.test_split, args.train_file_name, args.test_file_name)
