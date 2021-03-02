import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


if __name__ == "__main__":
    test_data = pd.read_csv("/home/yuliya/Desktop/ML_contest/2/test.csv")
    train_data = pd.read_csv("/home/yuliya/Desktop/ML_contest/2/train.csv")

    print(train_data.columns)

    y = train_data['critical_temperature']
    X = train_data.drop(['critical_temperature'], axis=1)

    forest_model = RandomForestRegressor(random_state=1)
    model = DecisionTreeRegressor(max_leaf_nodes=500, random_state=1)
    forest_model.fit(X, y)

    testX = test_data

    predicted = forest_model.predict(testX)

    f = open("answer.csv", "w")

    for i in range(len(predicted)):
        f.write(str(predicted[i]))
        if i < len(predicted) - 1:
            f.write('\n')

    f.close()
