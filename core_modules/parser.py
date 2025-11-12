"""
This script handles all data loading and preprocessing for the .arff files
provided for the assignment. It includes functions for:
- Loading .arff files into pandas DataFrames.
- Identifying numeric vs. categorical features.
- Imputing missing values (median for numeric, mode for categorical).
- Label encoding categorical features.
- Normalizing numeric features to a [0, 1] range.
- A main orchestrator to process all 10 folds for a given dataset.
"""

from scipy.io import arff
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Dict, Any, Optional

# --- Type Aliases for clarity ---
FoldData = Tuple[pd.DataFrame, pd.DataFrame]
ProcessedFold = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
EncoderDict = Dict[str, LabelEncoder]


def load_arff(filepath: str) -> pd.DataFrame:
    """
    Loads an .arff file from the given path into a pandas DataFrame.
    This uses the scipy.io.arff.loadarff function as suggested.

    Also decodes object-type columns (which scipy loads as bytes)
    into 'utf-8' strings.
    """
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)

    # Crucial: Decode byte strings (loaded by scipy) to native Python
    # strings for categorical features.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode('utf-8')

    return df


def get_class_column_name(df: pd.DataFrame) -> str:
    """
    Finds the name of the class column.
    Based on dataset structure, assumes the class is always the last column.
    The class must be stored for this assignment.
    """
    return df.columns[-1]


def identify_column_types(df: pd.DataFrame, class_column: str) -> Tuple[List[str], List[str]]:
    """
    Separates features into numeric and categorical lists,
    excluding the class column.
    This is essential for applying different preprocessing steps
    (e.g., normalization vs. encoding).
    """
    feature_columns = [col for col in df.columns if col != class_column]

    numeric_cols = []
    categorical_cols = []

    for col in feature_columns:
        if df[col].dtype in ['float64', 'int64']:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def handle_missing_values(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    """
    Imputes missing values in the DataFrame.
    - Numeric: Uses median, which is more robust to outliers than the mean.
    - Categorical: Uses mode (most frequent value).
    """
    df = df.copy()

    for col in numeric_cols:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    for col in categorical_cols:
        if df[col].isnull().any():
            # .mode()[0] selects the first mode if there are multiple
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)

    return df


def encode_categorical_variables(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_cols: List[str],
    class_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, EncoderDict]:
    """
    Applies LabelEncoder to all categorical features and the class column.
    It fits on the training data and transforms both train and test.

    Handles unseen categories in the test set by mapping them to -1.
    This is crucial to prevent errors during the transform phase.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    encoders: EncoderDict = {}

    # Encode features
    for col in categorical_cols:
        if col != class_column:  # Ensure we don't encode class col as a feature
            encoder = LabelEncoder()
            train_df[col] = encoder.fit_transform(train_df[col])

            # Justification: Handle unseen categories in the test set.
            # This is a critical step. We map unknown categories to -1
            # (or another outlier value) to avoid crashing the transform.
            test_categories = test_df[col]
            test_encoded = []
            for value in test_categories:
                if value in encoder.classes_:
                    test_encoded.append(encoder.transform([value])[0])
                else:
                    # Assign -1 to categories not seen in training
                    test_encoded.append(-1)
            test_df[col] = test_encoded

            encoders[col] = encoder

    # Encode class labels
    class_encoder = LabelEncoder()
    train_df[class_column] = class_encoder.fit_transform(train_df[class_column])

    # We assume the test set does not contain unseen *class labels*.
    # If it did, that would be a more significant dataset issue.
    test_df[class_column] = class_encoder.transform(test_df[class_column])
    encoders[class_column] = class_encoder

    return train_df, test_df, encoders


def normalize_numeric_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[MinMaxScaler]]:
    """
    Applies MinMaxScaler to scale features to the [0, 1] range.
    This is vital for distance-based algorithms (k-IBL) and SVMs.
    """
    if not numeric_cols:
        return train_df, test_df, None

    train_df = train_df.copy()
    test_df = test_df.copy()

    scaler = MinMaxScaler(feature_range=(0, 1))

    # Justification: Fit *only* on training data.
    # This prevents data leakage from the test set, ensuring the
    # normalization uses only information available during training.
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])

    # Transform test data using the *same* scaler (fit from train)
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    return train_df, test_df, scaler


def preprocess_fold(
    train_path: str,
    test_path: str,
    class_column: Optional[str] = None
) -> ProcessedFold:
    """
    Orchestrates the entire preprocessing pipeline for a single train/test fold.
    Returns the processed data as numpy arrays (TrainMatrix and TestMatrix).
    """
    # 1. Load data
    train_df = load_arff(train_path)
    test_df = load_arff(test_path)

    # 2. Identify class column
    if class_column is None:
        class_column = get_class_column_name(train_df)

    # 3. Identify feature types
    numeric_cols, categorical_cols = identify_column_types(train_df, class_column)

    # 4. Handle Missing Values
    train_df = handle_missing_values(train_df, numeric_cols, categorical_cols)
    test_df = handle_missing_values(test_df, numeric_cols, categorical_cols)

    # 5. Encode Categorical
    train_df, test_df, encoders = encode_categorical_variables(
        train_df, test_df, categorical_cols, class_column
    )

    # 6. Normalize Numeric
    # We only normalize the numeric columns, not the encoded categorical ones.
    train_df, test_df, scaler = normalize_numeric_features(
        train_df, test_df, numeric_cols
    )

    # 7. Finalize: Separate features (X) and target (y)
    X_train = train_df.drop(columns=[class_column]).values
    y_train = train_df[class_column].values

    X_test = test_df.drop(columns=[class_column]).values
    y_test = test_df[class_column].values

    return X_train, y_train, X_test, y_test


def load_all_folds(
    dataset_name: str,
    data_directory: str,
    class_column: Optional[str] = None
) -> List[ProcessedFold]:
    """
    Main function to load and preprocess all 10 folds for a given dataset.
    It expects a specific file naming convention:
    '{dataset_name}.fold.{fold_num:06d}.train.arff'

    The 10-fold CV sets are pre-defined and must be used as-is.
    """
    all_folds: List[ProcessedFold] = []

    # The assignment specifies a 10-fold CV
    for fold_num in range(10): # 0 through 9
        train_file = f"{dataset_name}.fold.{fold_num:06d}.train.arff"
        test_file = f"{dataset_name}.fold.{fold_num:06d}.test.arff"

        # Use relative paths by joining the base data directory
        train_path = os.path.join(data_directory, dataset_name, train_file)
        test_path = os.path.join(data_directory, dataset_name, test_file)

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"Warning: Files for fold {fold_num+1} not found. Skipping.")
            print(f"  Tried path: {train_path}")
            continue

        # Print statement to track cross-validation progress
        print(f"[Parser] Processing fold {fold_num+1}/10...")
        X_train, y_train, X_test, y_test = preprocess_fold(
            train_path, test_path, class_column
        )

        all_folds.append((X_train, y_train, X_test, y_test))

    print(f"\nSuccessfully loaded and processed {len(all_folds)} folds for dataset '{dataset_name}'.")
    return all_folds