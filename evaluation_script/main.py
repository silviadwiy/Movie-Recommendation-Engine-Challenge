import numpy as np
from sklearn.metrics import mean_squared_error


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    print("Submission related metadata:")
    num_columns = test_annotation_file.shape[1]
    rmse = np.zeros(num_columns)
    
    for i in range(num_columns):
        rmse[i] = mean_squared_error(test_annotation_file[:,i], user_submission_file[:,i], squared=False)
        
    mcrmse_score = np.mean(rmse)
    
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
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for Test Phase")
    return output
