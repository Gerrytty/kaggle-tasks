import pandas as pd


def get_list_of_categorical_variables(X_train):
    s = (X_train.dtypes == 'object')
    object_cols = list(s[s].index)

    print(f"Categorical variables: {object_cols}")

    return object_cols


def drop(X_train, X_valid):
    # Drop Categorical Variables
    drop_X_train = X_train.select_dtypes(exclude=['object'])
    drop_X_valid = X_valid.select_dtypes(exclude=['object'])

    return drop_X_train, drop_X_valid


def label_encoding(X_train, X_valid, object_cols):
    # Label Encoding
    from sklearn.preprocessing import LabelEncoder

    # Make copy to avoid changing original data
    label_X_train = X_train.copy()
    label_X_valid = X_valid.copy()

    # Apply label encoder to each column with categorical data
    label_encoder = LabelEncoder()

    for col in object_cols:
        label_X_train[col] = label_encoder.fit_transform(X_train[col])
        label_X_valid[col] = label_encoder.transform(X_valid[col])


def delete_unique_lables(X_train, X_valid):
    # All categorical columns
    object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

    # Columns that can be safely label encoded
    good_label_cols = [col for col in object_cols if set(X_train[col]) == set(X_valid[col])]

    # Problematic columns that will be dropped from the dataset
    bad_label_cols = list(set(object_cols) - set(good_label_cols))

    print('Categorical columns that will be label encoded:', good_label_cols)
    print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)


def one_hot_encoding(X_train, X_valid, object_cols):
    # One-Hot Encoding
    from sklearn.preprocessing import OneHotEncoder

    # Apply one-hot encoder to each column with categorical data
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

    # One-hot encoding removed index; put it back
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_valid = X_valid.drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

    return OH_X_train, OH_X_valid


def get_n_of_unique_entries_in_each_column(X_train, object_cols):
    # Get number of unique entries in each column with categorical data
    object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
    d = dict(zip(object_cols, object_nunique))

    # Print number of unique entries by column, in ascending order
    sorted(d.items(), key=lambda x: x[1])


def get_high_cardinality_cols(X_train, object_cols):
    # Columns that will be one-hot encoded
    low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

    # Columns that will be dropped from the dataset
    high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

    print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
    print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)

    return high_cardinality_cols


if __name__ == "__main__":
    arr1 = [1, 2, 3, 1, 1, 1]

    print(set(arr1))