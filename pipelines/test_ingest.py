from src.data.ingest import load_credit_data
from src.data.preprocess import handle_missing_values, standardize_column_names


def main():
    df = load_credit_data()

    df = standardize_column_names(df)
    df = handle_missing_values(df)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()
