import pandas as pd
from sklearn.metrics import roc_auc_score

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):

    # Load the ground truth labels and predictions into pandas dataframes
    ground_truth = pd.read_csv(test_annotation_file)
    prediction = pd.read_csv(user_submission_file)

    # Merge the two dataframes on the "id" and "word" columns
    merged_df = pd.merge(ground_truth, prediction, on='PhraseId')

    # Modify the sentiment labels and predicted scores for binary classification
    merged_df['actual_binary'] = merged_df['SentimentAja'].apply(lambda x: 1 if x in [4, 5] else 0)
    merged_df['predicted_binary'] = merged_df['Sentiment'].apply(lambda x: 1 if x >= 4 else 0)

    # Compute AUC-ROC score
    actual_binary = merged_df['actual_binary']
    predicted_binary = merged_df['predicted_binary']
    auc_roc = roc_auc_score(actual_binary, predicted_binary)



    output = {}

    if phase_codename == "dev":
        print("Evaluating for Dev Phase")
        output["result"] = [
            {
                "dev_split": {
                    'Score': auc_roc,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["dev_split"]
        print("Completed evaluation for Dev Phase")
    return output
