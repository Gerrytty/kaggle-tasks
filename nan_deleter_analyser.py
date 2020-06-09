import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error


file_path = ""
data = pd.read_csv(file_path)


def analyse(model, train_X, train_y, val_X, val_y):
    mae_with_drop, string_with_drop = model_with_drop(model, train_X, train_y, val_X, val_y)
    mae_with_immute, string_with_immute = immuted(model, train_X, train_y, val_X, val_y)
    mae_with_immute_x2, string_with_immute_x2 = immuted_x2(model, train_X, train_y, val_X, val_y)

    maes = [mae_with_drop, mae_with_immute, mae_with_immute_x2]
    strings = [string_with_drop, string_with_immute, string_with_immute_x2]

    print(f"The best way to delete null's val is {strings[maes.index(min(maes))]}")


def model_with_drop(model, train_X, train_y, val_X, val_y):
    # copy data for keeping consistency
    train_X_for_dropping = train_X.copy()
    val_X__for_dropping = val_X.copy()

    # Get names of columns with missing values
    cols_with_missing = [col for col in train_X_for_dropping.columns
                         if train_X_for_dropping[col].isnull().any()]

    # drop columns with missing parameters
    reduced_X_train = train_X_for_dropping.drop(cols_with_missing, axis=1)
    reduced_X_valid = val_X__for_dropping.drop(cols_with_missing, axis=1)

    model.fit(reduced_X_train, train_y)
    return mean_absolute_error(model.predict(reduced_X_valid), val_y), "model with drop columns"


def immuted(model, train_X, train_y, val_X, val_y):
    train_X_for_immute = train_X.copy()
    val_X_for_immute = val_X.copy()

    my_imputer = SimpleImputer()
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(train_X_for_immute))
    imputed_X_valid = pd.DataFrame(my_imputer.fit_transform(val_X_for_immute))

    imputed_X_train.columns = train_X_for_immute.columns
    imputed_X_valid.columns = val_X_for_immute.columns

    model.fit(imputed_X_train, train_y)
    return mean_absolute_error(model.predict(imputed_X_valid), val_y), "model with immute"


def immuted_x2(model, train_X, train_y, val_X, val_y):
    train_X_for_immute = train_X.copy()
    val_X_for_immute = val_X.copy()

    my_imputer = SimpleImputer()

    # Get names of columns with missing values
    cols_with_missing = [col for col in train_X_for_immute.columns
                         if train_X_for_immute[col].isnull().any()]

    # Make new columns indicating what will be imputed
    for col in cols_with_missing:
        train_X_for_immute[col + '_was_missing'] = train_X_for_immute[col].isnull()
        val_X_for_immute[col + '_was_missing'] = val_X_for_immute[col].isnull()

    imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(train_X_for_immute))
    imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(val_X_for_immute))

    # Imputation removed column names; put them back
    imputed_X_train_plus.columns = train_X_for_immute.columns
    imputed_X_valid_plus.columns = val_X_for_immute.columns

    model.fit(imputed_X_train_plus, train_y)
    return mean_absolute_error(model.predict(imputed_X_valid_plus), val_y), \
           "more complex model with immute"


if __name__ == "__main__":
    pass