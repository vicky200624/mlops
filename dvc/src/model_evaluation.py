import numpy as np
import pandas as pd
import pickle
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score

try:
    # Load model
    clf = pickle.load(open('models/model.pkl','rb'))

    # Load test data
    test_data = pd.read_csv('./data/features/test_bow.csv')

    X_test = test_data.iloc[:,0:-1].values
    y_test = test_data.iloc[:,-1].values

    # Predictions
    y_pred = clf.predict(X_test)

    # Metrics for multiclass
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics_dict, file, indent=4)

    print("Model evaluation completed successfully")

except Exception as e:
    print("Error during model evaluation")
    print(e)