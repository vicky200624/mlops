import numpy as np
import pandas as pd
import pickle
import os

from sklearn.ensemble import GradientBoostingClassifier

try:
    # Load feature data
    train_data = pd.read_csv('./data/features/train_bow.csv')

    # Split features and label
    X_train = train_data.iloc[:, 0:-1].values
    y_train = train_data.iloc[:, -1].values

    # Train model
    clf = GradientBoostingClassifier(n_estimators=50)
    clf.fit(X_train, y_train)

    # Create models directory
    model_path = "models"
    os.makedirs(model_path, exist_ok=True)

    # Save model
    with open(os.path.join(model_path, "model.pkl"), "wb") as file:
        pickle.dump(clf, file)

    print("Model training completed and saved successfully")

except Exception as e:
    print("Error during model training")
    print(e)