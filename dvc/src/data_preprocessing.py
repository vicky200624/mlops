import numpy as np
import pandas as pd
import os
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load data
train_data = pd.read_csv('./data/raw/train.csv')
test_data = pd.read_csv('./data/raw/test.csv')

# Download NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')


def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    words = [i for i in str(text).split() if i not in stop_words]
    return " ".join(words)


def removing_numbers(text):
    return ''.join([i for i in text if not i.isdigit()])


def lower_case(text):
    words = text.split()
    words = [w.lower() for w in words]
    return " ".join(words)


def removing_punctuations(text):
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)


def normalize_text(df):

    df["Ticket_Description"] = df["Ticket_Description"].astype(str)

    df["Ticket_Description"] = df["Ticket_Description"].apply(lower_case)
    df["Ticket_Description"] = df["Ticket_Description"].apply(remove_stop_words)
    df["Ticket_Description"] = df["Ticket_Description"].apply(removing_numbers)
    df["Ticket_Description"] = df["Ticket_Description"].apply(removing_punctuations)
    df["Ticket_Description"] = df["Ticket_Description"].apply(removing_urls)
    df["Ticket_Description"] = df["Ticket_Description"].apply(lemmatization)

    return df


# Apply preprocessing
train_processed_data = normalize_text(train_data)
test_processed_data = normalize_text(test_data)


# Save processed data
data_path = os.path.join("data", "processed")

os.makedirs(data_path, exist_ok=True)

train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

print("Data preprocessing completed successfully")