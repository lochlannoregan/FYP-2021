# !/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_curve, \
    plot_precision_recall_curve, average_precision_score
import shap
import matplotlib.pyplot as plt


def import_data():
    return pd.read_csv("data/bank/bank.csv", sep=";")


def pre_processing(data):
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['y_encoded'], axis=1), data['y_encoded'])
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


# Data pre-processing / cleaning etc. - data preparation is right name??
def data_preparation(data):
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

    return data_numeric


def mlp_model(X_train, y_train, X_test, y_test):

    # # Apply Standard scaling to get better results
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    # sklearn neural network
    # mlpc = MLPClassifier(learning_rate_init=0.07, activation='logistic', hidden_layer_sizes=5, max_iter=500)
    mlpc = MLPClassifier()
    mlpc.fit(X_train, y_train)

    # accuracy = str(accuracy_score(y_test, pred_mlpc) * 100)

    # # print the models performance
    # print("\t\tAccuracy: " + accuracy)
    # print("Accuracy: " + accuracy + "\n")
    # print(str(classification_report(y_test, pred_mlpc)) + "\n")
    # print(str(confusion_matrix(y_test, pred_mlpc)) + "\n\n")
    # output_file.write("Accuracy: " + accuracy + "\n")
    # output_file.write(str(classification_report(y_test, pred_mlpc)) + "\n")
    # output_file.write(str(confusion_matrix(y_test, pred_mlpc)) + "\n\n")

    return mlpc


def model_performance(mlpc, X_test, y_test):
    pred_mlpc = mlpc.predict(X_test)
    print(str(classification_report(y_test, pred_mlpc)) + "\n")
    print(str(confusion_matrix(y_test, pred_mlpc)) + "\n\n")


def shap_usage(X_train, X_test, mlpc):
    explainer = shap.KernelExplainer(mlpc.predict_proba, X_train)
    shap_values = explainer.shap_values(X_test.iloc[0,:])
    shap.force_plot(explainer.expected_value[0], shap_values[0], X_test.iloc[0,:], matplotlib=True)

    # shap_values = explainer.shap_values(X_test)
    # shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)


def main():
    shap.initjs()

    data = import_data()
    data_numeric = data_preparation(data)

    print(data_numeric.head())

    X_train, X_test, y_train, y_test = pre_processing(data_numeric)

    mlpc = mlp_model(X_train, y_train, X_test, y_test)

    model_performance(mlpc, X_test, y_test)

    shap_usage(X_train, X_test, mlpc)

    print("Testing line")


if __name__ == '__main__':
    main()
