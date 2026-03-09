import pandas as pd
import os
from sklearn.model_selection import train_test_split
import kagglehub


def load_data() -> pd.DataFrame:
    try:
        # Download dataset
        dataset_path = kagglehub.dataset_download(
            "ajverse/customer-support-tickets-crm-dataset"
        )

        # Load CSV file
        df = pd.read_csv(dataset_path + "/customer_support_tickets.csv")

        print("Dataset loaded successfully")
        print("Columns:", df.columns)

        return df

    except Exception as e:
        print("Error while loading dataset")
        print(e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Remove ticket id column if exists
        if "Ticket ID" in df.columns:
            df = df.drop(columns=["Ticket ID"])

        # Drop rows with missing values
        df = df.dropna()

        return df

    except Exception as e:
        print("Error during preprocessing")
        print(e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        raw_path = os.path.join(data_path, "raw")
        os.makedirs(raw_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_path, "test.csv"), index=False)

        print("Train and test data saved successfully")

    except Exception as e:
        print("Error while saving data")
        print(e)
        raise


def main():
    try:
        df = load_data()
        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(
            final_df, test_size=0.2, random_state=42
        )

        save_data(train_data, test_data, data_path="data")

    except Exception as e:
        print(f"Error: {e}")
        print("Failed to complete the data ingestion process.")


if __name__ == "__main__":
    main()