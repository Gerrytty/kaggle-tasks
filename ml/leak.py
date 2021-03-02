from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

if __name__ == "__main__":
    my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
    cv_scores = cross_val_score(my_pipeline, X, y,
                                cv=5,
                                scoring='accuracy')

    print("Cross-validation accuracy: %f" % cv_scores.mean())

    expenditures_cardholders = X.expenditure[y]
    expenditures_noncardholders = X.expenditure[~y]

    print('Fraction of those who did not receive a card and had no expenditures: %.2f' % ((expenditures_noncardholders == 0).mean()))
    print('Fraction of those who received a card and had no expenditures: %.2f' % ((expenditures_cardholders == 0).mean()))