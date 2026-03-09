import numpy as np
import pandas as pd
import os

from sklearn.feature_extraction.text import CountVectorizer

# Load processed data
train_data = pd.read_csv('./data/processed/train_processed.csv')
test_data = pd.read_csv('./data/processed/test_processed.csv')

train_data.fillna('', inplace=True)
test_data.fillna('', inplace=True)

# Text column
X_train = train_data['Ticket_Description'].values
X_test = test_data['Ticket_Description'].values

# Target column
y_train = train_data['Satisfaction_Score'].values
y_test = test_data['Satisfaction_Score'].values

# Bag of Words
vectorizer = CountVectorizer(max_features=50)

# Fit on training data
X_train_bow = vectorizer.fit_transform(X_train)

# Transform test data
X_test_bow = vectorizer.transform(X_test)

# Convert to dataframe
train_df = pd.DataFrame(X_train_bow.toarray())
train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())
test_df['label'] = y_test

# Save features
data_path = os.path.join("data", "features")
os.makedirs(data_path, exist_ok=True)

train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)

print("Feature engineering completed successfully")