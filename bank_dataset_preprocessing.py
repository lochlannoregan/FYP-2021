import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib


def load_and_preprocess():
    data = pd.read_csv("data/bank-additional/bank-additional.csv", sep=";")

    numerical_features = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                          'euribor3m', 'nr.employed', 'duration']
    categorical_features_nominal = ['job', 'marital', 'contact', 'month', 'day_of_week', 'poutcome']
    categorical_features_binary = ['default', 'housing', 'loan']
    categorical_features_unknown_values = ['education', 'default', 'housing', 'loan', 'job', 'marital']

    X = data.drop(['y'], axis=1)
    y = data['y']

    education_ordinal_mapping = {'illiterate': 0, 'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3, 'high.school': 4,
                                 'professional.course': 5,
                                 'university.degree': 6}
    binary_features_mapping = {'yes': 1, 'no': 0}

    simple_imputer = SimpleImputer(missing_values="unknown", strategy="most_frequent")
    simple_imputer.fit(X[categorical_features_unknown_values])
    X[categorical_features_unknown_values] = simple_imputer.transform(X[categorical_features_unknown_values])

    X['education'] = X['education'].replace(education_ordinal_mapping)

    X[categorical_features_binary] = X[categorical_features_binary].replace(binary_features_mapping)

    y = y.replace(binary_features_mapping)

    # Scaling
    scaler = StandardScaler()
    scaler.fit(X[numerical_features])
    X[numerical_features] = scaler.transform(X[numerical_features])

    X_dummies = pd.get_dummies(X[categorical_features_nominal], drop_first=True)
    X = X.drop(categorical_features_nominal, axis='columns')
    X = pd.concat([X, X_dummies], axis='columns')

    # print(y_train.describe())
    # X_train, y_train = SMOTE(random_state=0).fit_resample(X_train, y_train)
    # print(y_train.describe())

    joblib.dump(X, "X.joblib")
    joblib.dump(y, "y.joblib")

    return X, y


if __name__ == "__main__":
    load_and_preprocess()