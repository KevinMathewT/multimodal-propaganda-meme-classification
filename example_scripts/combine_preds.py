import pandas as pd
import numpy as np
import json
from sklearn.metrics import f1_score

# Function to load predictions
def load_predictions(files):
    dfs = [pd.read_csv(file, sep='\t') for file in files]
    return dfs

# Function to load actual labels from JSON
def load_actual_labels(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Extract only the relevant columns
    relevant_data = [{'id': item['id'], 'class_label': item['class_label']} for item in data]
    labels_df = pd.DataFrame(relevant_data)
    return labels_df

# Majority Voting
def majority_voting(dfs):
    binary_predictions = [df['prob'].apply(lambda x: 'propaganda' if x > 0.5 else 'not_propaganda') for df in dfs]
    majority_vote = pd.concat(binary_predictions, axis=1).mode(axis=1)[0]
    result = dfs[0][['id']].copy()
    result['label'] = majority_vote
    return result

# Average Probability
def average_probability(dfs):
    average_prob = pd.concat([df[['id', 'prob']] for df in dfs]).groupby('id').mean().reset_index()
    return average_prob

# Threshold Optimization
def threshold_optimization(df, labels_df):
    def find_optimal_threshold(y_true, y_prob):
        thresholds = np.linspace(0, 1, 100)
        f1_scores = [f1_score(y_true, y_prob > t) for t in thresholds]
        # print(y_true)
        # print("*****")
        # print(y_prob)
        # print("*****")
        # print(thresholds)
        # print("*****")
        # print(f1_scores)
        print(thresholds[np.argmax(f1_scores)], f1_scores[np.argmax(f1_scores)])
        print("*****")
        return thresholds[np.argmax(f1_scores)]
    
    # Merge the actual labels with the predicted probabilities on 'id'
    merged_df = pd.merge(df, labels_df, on='id', how='left')
    
    # Convert 'class_label' to binary
    y_true = merged_df['class_label'].apply(lambda x: 1 if x == 'propaganda' else 0).values
    y_prob = merged_df['prob'].values
    
    # Find the optimal threshold
    optimal_threshold = find_optimal_threshold(y_true, y_prob)
    
    # Apply the optimal threshold to make final predictions
    merged_df['label'] = merged_df['prob'].apply(lambda x: 'propaganda' if x > optimal_threshold else 'not_propaganda')
    
    # Return only the original columns from df with updated 'label'
    return merged_df[['id', 'prob', 'label']]

# Example usage
files = [f'task2C_kevinmathew_probs_fold_{i}.tsv' for i in range(5)]
dfs = load_predictions(files)
for f in dfs:
    print(f['id'].tolist()[0], f['prob'].tolist()[0])
json_file = 'data/arabic_memes_propaganda_araieval_24_dev.json'
labels_df = load_actual_labels(json_file)

# Chain functions
average_df = average_probability(dfs)
for df in dfs:
    threshold_optimization(df, labels_df)

threshold_optimized_df = threshold_optimization(average_df, labels_df)

# print(average_df)
# print(threshold_optimized_df)

# Save results
# average_df.to_csv('average_probability_result.tsv', sep='\t', index=False)
# threshold_optimized_df.to_csv('threshold_optimization_result.tsv', sep='\t', index=False)