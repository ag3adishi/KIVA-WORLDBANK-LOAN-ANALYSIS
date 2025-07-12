from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def prepare_data(df, target_column):
    # Drop non-useful columns
    drop_cols = [
        'id', 'tags', 'use', 'region', 'posted_time',
        'disbursed_time', 'funded_time', 'date'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe!")

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # DEBUG: See what types are left
    print("X.dtypes BEFORE encoding:")
    print(X.dtypes)

    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # DEBUG: Final check after encoding
    print("\nX.dtypes AFTER encoding:")
    print(X.dtypes)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Keep only numeric (double safe)
    X_train = pd.DataFrame(X_train).select_dtypes(include='number')
    X_test = pd.DataFrame(X_test).select_dtypes(include='number')

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler




