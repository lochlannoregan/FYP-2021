# !/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer


def import_data():
    return pd.read_csv("data/bank/bank.csv", sep=";")


def training_testing_split(data):
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['y'], axis=1), data['y'])
    return X_train, X_test, y_train, y_test


# Data pre-processing / cleaning etc. - data preparation is right name??
def pre_processing(data):
    # data = pd.get_dummies(data, columns=['job', 'marital', 'education', 'default', 'housing',
    #                                      'loan', 'contact', 'month', 'poutcome', 'y'])
    # data['Test'] = pd.factorize(data.job)[0]
    # print(data.job.value_counts())
    # print(data.Test.value_counts())

    le = LabelEncoder()

    data['job_encoded'] = le.fit_transform(data.job)
    data['marital_encoded'] = le.fit_transform(data.marital)
    data['education_encoded'] = le.fit_transform(data.education)
    data['default_encoded'] = le.fit_transform(data.default)
    data['housing_encoded'] = le.fit_transform(data.housing)
    data['loan_encoded'] = le.fit_transform(data.loan)
    data['contact_encoded'] = le.fit_transform(data.contact)
    data['month_encoded'] = le.fit_transform(data.month)
    data['poutcome_encoded'] = le.fit_transform(data.poutcome)
    data['y_encoded'] = le.fit_transform(data.y)

    data_numeric = data[['age', 'job_encoded', 'marital_encoded', 'education_encoded', 'default_encoded', 'balance',
                         'housing_encoded', 'loan_encoded', 'contact_encoded', 'day', 'month_encoded',
                         'duration', 'campaign', 'pdays', 'previous', 'poutcome_encoded', 'y_encoded']].copy()

    print("finished")
    return data_numeric


def main():
    data = import_data()
    data_numeric = pre_processing(data)

    print(data_numeric.head())

    # X_train, X_test, y_train, y_test = training_testing_split(data)



    # X_train, X_test, y_train, y_test = pre_processing(X_train, X_test, y_train, y_test)
    # linear_clf = linear_classification_training(X_train, y_train)
    # decision_tree_clf = decision_tree(X_train, y_train)
    # print(accuracy_score(y_test, linear_clf.predict(X_test)))


if __name__ == '__main__':
    main()
