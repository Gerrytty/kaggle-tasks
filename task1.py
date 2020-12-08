import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# OFFSHORE = 0
# ONSHORE = 1


def preprocess(df, name, le):
    le.fit(df[name].unique())
    df[name] = le.transform(df[name])


if __name__ == "__main__":
    test_data = pd.read_csv("test.csv")
    train_data = pd.read_csv("train.csv")

    # print(train_data.columns)

    features = ['Tectonic regime', 'Hydrocarbon type', 'Structural setting', 'Reservoir status',
                'Depth', 'Period', 'Gross', 'Lithology', 'Netpay', 'Porosity', 'Permeability']

    le = LabelEncoder()

    for feature in features:
        preprocess(train_data, feature, le)

    for feature in features:
        preprocess(test_data, feature, le)

    le.fit(train_data['Onshore/Offshore'].unique())
    train_data['Onshore/Offshore'] = le.transform(train_data['Onshore/Offshore'])

    y = train_data['Onshore/Offshore']
    X = train_data[features]

    forest_model = RandomForestRegressor(random_state=2)
    model = DecisionTreeRegressor(max_leaf_nodes=500, random_state=2)
    forest_model.fit(X, y)

    testX = test_data[features]

    predicted = forest_model.predict(testX)

    print(predicted)
    print(y)

    arr = []

    f = open("prediction.csv", "w")

    for i in range(len(predicted)):
        if predicted[i] <= 0.5:
            ans = 'OFFSHORE'
        else:
            ans = 'ONSHORE'

        f.write(ans)
        if i < len(predicted) - 1:
            f.write('\n')
        arr.append(ans)

    f.close()

    d = pd.read_csv("prediction.csv")
    print(d)
