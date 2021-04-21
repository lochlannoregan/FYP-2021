import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

    X_dummies = pd.get_dummies(X[categorical_features_nominal], drop_first=True)
    X = X.drop(categorical_features_nominal, axis='columns')
    X = pd.concat([X, X_dummies], axis='columns')

    # Incorrect ordinal encoding of categorical nominal features
    # for feature in categorical_features_nominal:
    #     unique_values = X[feature].unique()
    #     feature_dict_mapping = {}
    #     for i, value in enumerate(unique_values):
    #         feature_dict_mapping[value] = i
    #     X[feature] = X[feature].replace(feature_dict_mapping)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train.loc[:, numerical_features])
    X_train.loc[:, numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
    X_test.loc[:, numerical_features] = scaler.transform(X_test.loc[:, numerical_features])

    X.loc[:, numerical_features] = scaler.transform(X.loc[:, numerical_features])

    joblib.dump(X, "saved-model-computation/X.joblib")
    joblib.dump(y, "saved-model-computation/.joblib")
    joblib.dump(X_train, "saved-model-computation/X_train.joblib")
    joblib.dump(X_test, "saved-model-computation/X_test.joblib")
    joblib.dump(y_train, "saved-model-computation/y_train.joblib")
    joblib.dump(X_test, "saved-model-computation/y_test.joblib")

    return X, y, X_train, X_test, y_train, y_test


def load_preprocessed_data():
    X = joblib.load("saved-model-computation/X.joblib")
    y = joblib.load("saved-model-computation/y.joblib")
    X_train = joblib.load("saved-model-computation/X_train.joblib")
    X_test = joblib.load("saved-model-computation/X_test.joblib")
    y_train = joblib.load("saved-model-computation/y_train.joblib")
    y_test = joblib.load("saved-model-computation/y_test.joblib")
    return X, y, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    load_and_preprocess()