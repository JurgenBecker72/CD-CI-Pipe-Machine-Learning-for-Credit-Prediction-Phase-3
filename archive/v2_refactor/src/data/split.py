# src/data/split.py
# Train / validation / test split with sensible defaults.
# Default target was "default" in the original — changed to the real target.

from sklearn.model_selection import train_test_split

from src.config import TARGET, RANDOM_STATE, TEST_SIZE, VAL_SIZE


def split_data(
    df,
    target_col: str = TARGET,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First peel off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Then carve validation out of the remainder so that val_size is
    # expressed relative to the full dataset, not relative to temp.
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_relative,
        stratify=y_temp,
        random_state=random_state,
    )

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test
