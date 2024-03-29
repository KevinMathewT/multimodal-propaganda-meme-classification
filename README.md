# Task 2: Multimodal Propagandistic Memes Classification

The objective of this task is to categorize memes as either propagandistic or non-propagandistic. For more details, please refer to the [Task Description](#task-description) provided below.

__Table of contents:__
<!-- - [Evaluation Results](#evaluation-results) -->
- [List of Versions](#list-of-versions)
- [Contents of the Directory](#contents-of-the-directory)
- [Task Description](#task-escription)
- [Dataset](#dataset)
- [Scorer and Official Evaluation Metrics](#scorer-and-official-evaluation-metrics)
- [Baselines](#baselines)<!-- - [Submission guidelines](#submission-guidelines) -->
- [Format Checkers](#format-checkers)
- [Submission](#submission)
- [Credits](#credits)



<!-- ## Evaluation Results
Submitted results will be available after the system submission deadline.
Kindly find the leaderboard released in this google sheet, [link](http://shorturl.at/nuCOS). you can find in the tab labeled "Task 1".
 -->


## List of Versions

* __Task 2 [2024/03/16]__
  - Training/Dev data for task 2 released.


## Contents of the Directory

* Main folder: [data](./data)
  	This directory contains files for all languages and subtasks.
* Main folder: [baselines](./baselines)<br/>
	Contains scripts provided for baseline models of the tasks.
* Main folder: [format_checker](./format_checker)<br/>
	Contains scripts provided to check format of the submission file.
* Main folder: [scorer](./scorer)<br/>
	Contains scripts provided to score output of the model when provided with label (i.e., dev set).

* [README.md](./README.md) <br/>
	This file!

## Task Description
For the multimodal propagandistic memes classification task, we offer the three subtasks, which are defined below:

- Subtask 2A: Given a text extracted from meme, categorize whether it is propagandistic or not.
- Subtask 2B: Given a meme (text overlayed image), the task is to detect whether the content is propagandistic.
- Subtask 2C: Given multimodal content (text extracted from meme and the meme itself) the task is to detect whether the content is propagandistic.

All are binary classification tasks.


## Dataset
All datasets are JSONL files where each line has one JSON object. The text encoding is UTF-8. Images are in different directory with a pointer in the join object. The input and the result files have the same format for all the subtasks.

#### Input Data Format

We have two splits train, dev jsonl files. Each file has lines of dictionary objects containing the id, text, labels, and other information. The text encoding is UTF-8. Each dictionary object has the following keys:
`id`, `img_path`, `text`, `class_label`


All the images are segregated under the folders `arabic_memes_fb_insta_pinterest`, `fb_memes`, and `images`.

**Examples:**
``` 
{
	"id": "data/arabic_memes_fb_insta_pinterest/Instagram/IMAGES/ex.officiall/2019-10-25_17-08-21_UTC.jpg", 
	"img_path": "data/arabic_memes_fb_insta_pinterest/Instagram/IMAGES/ex.officiall/2019-10-25_17-08-21_UTC.jpg", 
	"text": "- انا من حقي اقول اني مبحبش الشتا \n= وانا من حقي اهينك", 
	"class_label": "not_propaganda"
},
{
	"id": "data/arabic_memes_fb_insta_pinterest/Pinterest/images/pinterest_images_part2/www.pinterest.com_pin_302163456262798283/add7d8d70902628fb3ae0dff1fb5568b.jpg", 
	"img_path": "data/arabic_memes_fb_insta_pinterest/Pinterest/images/pinterest_images_part2/www.pinterest.com_pin_302163456262798283/add7d8d70902628fb3ae0dff1fb5568b.jpg", 
	"text": "أنا مش هضعف تاني قصادك أنا مش هرجع ابص ورايا..حنفيييي\n خلااااص اديني رجعتلك اديني بين ايديكي", 
	"class_label": "not_propaganda"
},
{ 
	"id": "data/arabic_memes_fb_insta_pinterest/Instagram/IMAGES/ex.officiall/2021-04-08_09-56-41_UTC.jpg", 
	"img_path": "data/arabic_memes_fb_insta_pinterest/Instagram/IMAGES/ex.officiall/2021-04-08_09-56-41_UTC.jpg", 
	"text": "-لما الكراش يتجاهلني\n=..\nMe\nده باين عليه بيحبني\nآه شكله بيحبني بس بيتقل", 
	"class_label": "not_propaganda"
}
```

### Output Data Format
For all subtasks **2A**, **2B**, and **2C** the submission files format is the same.

The expected results file is a list of tweets/transcriptions with the predicted class label.

The file header should strictly be as follows:

> **id <TAB> class_label <TAB> run_id**

Each row contains three TAB separated fields:

> id <TAB> class_label <TAB> run_id

Where: <br>
* id: id for a given data<br/>
* class_label: Predicted class label for the tweet. <br/>
* run_id: String identifier used by participants. <br/>

Example:
```
data/arabic_memes_fb_insta_pinterest/Instagram/IMAGES/ex.officiall/2019-10-25_17-08-21_UTC.jpg	not_propaganda  Model_1
data/arabic_memes_fb_insta_pinterest/Pinterest/images/pinterest_images_part2/www.pinterest.com_pin_302163456262798283/add7d8d70902628fb3ae0dff1fb5568b.jpg	not_propaganda  Model_1
data/arabic_memes_fb_insta_pinterest/Facebook/images/ArabianMemez/339315940_3510069309314547_8940591235150933516_n.jpg	propaganda  Model_1
```

## Scorer and Official Evaluation Metrics

### Scorers
The scorer for the task is located in the [scorer](./scorer) module of the project.
To launch the script you need to install packages dependencies found in [requirements.txt](./requirements.txt) using the following:
> pip3 install -r requirements.txt <br/>

Launch the scorer for the subtask as follows:
> python scorer/task2.py --gold-file-path=<path_gold_file> --pred-file-path=<predictions_file> <br/>

The scorer invokes the format checker for the task to verify the output is properly shaped.
It also handles checking if the provided predictions file contains all tweets from the gold one.

##### Example
<!--python scorer/task2.py --pred-file-path=task1_dev_output.jsonl --gold-file-path data/task1_dev.jsonl-->

```
python scorer/task2.py --gold-file-path data/arabic_memes_propaganda_araieval_24_dev.json --pred-file-path data/majority_baseline_arabic_memes_propaganda_araieval_24_dev.tsv 
```


### Official Evaluation Metrics
The **official evaluation metric** for all subtasks is **macro-F1**. However, the scorer also reports macro-F1, Precision, and Recall.
s

## Baselines

The [baselines](./baselines) module currently contains a majority and random baseline for subtask 2A, subtask 2B, and subtask 2C. Additionally, a simple n-gram baseline for subtask 2A and subtask 2C, an unoptimized simple baseline of SVM over Image feature for 2B, and an unoptimized simple baseline of SVM over concatenated Image and Text features is added for Subtask 2C.

**Baseline Results on the Dev set**

|Model|subtask-2A|subtask-2B|subtask-2C|
|:----|:----|:----|:----|
|Random Baseline |0.466|0.471|0.518|
|Majority Baseline|0.418|0.418|0.418|
|n-gram Baseline|0.579|NA|0.579|
|ResNet SVM|NA|0.617|NA|
|Multimodal:<br/>ResNet+BERT SVM|NA|NA|0.695|


To launch the baseline script you need to install packages dependencies found in [requirements.txt](./requirements.txt) using the following:
> pip3 install -r requirements.txt <br/>

### Subtask 2A
To launch the baseline script run the following:
> python baselines/subtask_2a.py --train-file-path=<path_to_your_training_file> --dev-file-path=<path_of_your_dev_file><br/>

<!--```python baselines/subtask_2a.py --data-dir=data/ --train-file-path=arabic_memes_propaganda_araieval_24_train.json --dev-file-path=arabic_memes_propaganda_araieval_24_dev.json```-->
```
python baselines/subtask_2a.py --train-file-path data/arabic_memes_propaganda_araieval_24_train.json --dev-file-path data/arabic_memes_propaganda_araieval_24_dev.json
```

### Subtask 2B
To extract the Image features, run the following script:
> python baseline/extract_feat.py --data-dir ./data --file-name <path_to_your_training_file> --output-file-name <path_to_image_features_file> 
<!--```python baseline/extract_feat.py --data-dir ./ --file-name arabic_memes_propaganda_araieval_24_train.json --output-file-name train_feats.json```-->
```
python baselines/extract_feat.py --data-dir data/ --file-name arabic_memes_propaganda_araieval_24_train.json --out-file-name train_feats.json
```

To launch the baseline script run the following:
> python baselines/subtask_2b.py --data-dir=<data-directory> --test-split=<split-name> --train-file-name=<name_of_your_training_file> --test-file-name=<name_of_your_test_file_to_be_evaluated><br/>
<!--```python3 baselines/subtask_2b.py --data-dir=data/ --test-split=dev --train-file-name=arabic_memes_propaganda_araieval_24_train.json --test-file-name=arabic_memes_propaganda_araieval_24_dev.json```-->
```
python baselines/subtask_2b.py --data-dir=data/ --test-split=dev --train-file-name=arabic_memes_propaganda_araieval_24_train.json --test-file-name=arabic_memes_propaganda_araieval_24_dev.json
```

### Subtask 2C
To extract the Image and text features, run the following script:
> python baseline/extract_feat.py --data-dir ./data --file-name arabic_memes_propaganda_araieval_24_train.json --output-file-name train_feats.json
```
python baseline/extract_feat.py --data-dir ./ --file-name arabic_memes_propaganda_araieval_24_train.json --output-file-name train_feats.json
```

To launch the baseline script run the following:
> python3 baselines/subtask_2c.py --data-dir=<data-directory> --test-split=<split-name> --train-file-name=<name_of_your_training_file> --test-file-name=<name_of_your_test_file_to_be_evaluated><br/>
```
python baselines/subtask_2c.py --data-dir=data/ --test-split=dev --train-file-name=arabic_memes_propaganda_araieval_24_train.json --test-file-name=arabic_memes_propaganda_araieval_24_dev.json
```

All baselines will be trained on the training dataset and the performance of the model is evaluated on the dev set.

## Format Checker

The checker for the task is located in the [format_checker](./format_checker) module of the project.
To launch the checker script you need to install packages dependencies found in [requirements.txt](./requirements.txt) using the following:
> pip3 install -r requirements.txt <br/>

The format checker verifies that your generated results files complies with the expected format.
To launch it run:

> python3 format_checker/task2.py --pred-files-path <path_to_result_file_1 path_to_result_file_2 ... path_to_result_file_n> <br/>

`--pred-files-path` is to be followed by a single string that contains a space separated list of one or more file paths.

__<path_to_result_file_n>__ is the path to the corresponding file with participants' predictions, which must follow the format, described in the [Output Data Format](#output-data-format) section.

Note that the checker can not verify whether the prediction files you submit contain all tweets, because it does not have access to the corresponding gold file.

## Submission

### Guidelines
The process consists of two phases:

1. **System Development Phase:** This phase involves working on the dev set.
2. **Final Evaluation Phase (will start on 27 April 2023):** This phase involves working on the test set, which will be released during the evaluation cycle.

For each phase, please adhere to the following guidelines:
- Each team should create and maintain a single account for submissions. Please ensure all runs are submitted through this account. Submissions from multiple accounts by the same team could result in your system being not ranked in the overview paper.
- The most recent file submitted to the leaderboard will be considered your final submission.
- The output file must be named task[2][A/B/C]_any_suffix.tsv, where [2] is the task number and A/B/C is your specific subtask (for example, task2A_team_name.tsv or task2B_team_name.tsv). Failure to follow this naming convention will result in an error on the leaderboard. Subtasks include 2A, 2B and 2C.
- You are required to compress the .tsv file into a .zip file (for example, zip task2A.zip task2A.tsv) and submit it via the Codalab page.
- Please include your team name and a description of your method with each submission.
- You are permitted to submit a maximum of 200 submissions per day for each subtask.

### Submission Site
Please submit your results on the respective subtask tab: https://codalab.lisn.upsaclay.fr/competitions/18099


## Licensing
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. This allows for sharing and adapting the work, provided that attribution is given, the use is non-commercial, and any derivative works are shared under the same terms. For more information, please visit https://creativecommons.org/licenses/by-nc-sa/4.0/.


## Credits
Please find it on the task website: https://araieval.gitlab.io/task2/
# multimodal-propaganda-meme-classification
