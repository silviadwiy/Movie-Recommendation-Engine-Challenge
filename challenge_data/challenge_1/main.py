import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    print("Submission related metadata:")

    df1 = pd.read_csv(test_annotation_file, index_col='text_id')
    df2 = pd.read_csv(user_submission_file, index_col='text_id')

    # Extract the target variables from each data frame
    y_true = df1[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']]
    y_pred = df2[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']]

    # Calculate MCRMSE
    mcrmse_score = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values')).mean()

    print(kwargs["submission_metadata"])
    output = {}
    if phase_codename == "dev":
        print("Evaluating for Dev Phase")
        output["result"] = [
            {
                "train_split": {
                    "Score": mcrmse_score,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["train_split"]
        print("Completed evaluation for Dev Phase")
    return output
