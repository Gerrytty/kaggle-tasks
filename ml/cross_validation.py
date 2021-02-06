from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def get_score(n_estimators, X, y):
    """Return the average MAE over 3 CV folds of random forest model.

    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    # Replace this body with your own code
    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                  ('model', RandomForestRegressor(n_estimators=n_estimators,
                                                                  random_state=0))
                                  ])

    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')

    return scores.mean()


if __name__ == "__main__":
    results = {result: get_score(result, X, y) for result in range(50, 450, 50)}  # Your code here

    import matplotlib.pyplot as plt

    plt.plot(list(results.keys()), list(results.values()))
    plt.show()

    # define a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                                  ('model', RandomForestRegressor(n_estimators=50,
                                                                  random_state=0))
                                  ])

    # cross validation
    from sklearn.model_selection import cross_val_score

    # Multiply by -1 since sklearn calculates *negative* MAE
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=5,
                                  scoring='neg_mean_absolute_error')

    print("MAE scores:\n", scores)