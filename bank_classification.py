# !/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
import scipy as sc
from imblearn.over_sampling import SMOTE
from collections import Counter


def import_data():
    return pd.read_csv("data/bank/bank.csv", sep=";")


def pre_processing(data):
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['y_encoded'], axis=1), data['y_encoded'])
    print(Counter(y_train).items())
    X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)
    print(Counter(y_train).items())
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # scaler.transform(X_train)
    # scaler.transform(X_test)

    return X_train_resampled, X_test, y_train_resampled, y_test


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

    # data_numeric = data[['age', 'default_encoded', 'balance', 'housing_encoded', 'loan_encoded', 'duration', 'campaign', 'pdays', 'previous', 'poutcome_encoded', 'y_encoded']].copy()

    return data_numeric


def mlp_model(X_train, y_train, X_test, y_test):

    # # Apply Standard scaling to get better results
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    # pipeline = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=8))
    # #
    # # Create the randomized search estimator
    # #
    # param_distributions = [{'mlp__C': sc.stats.expon(scale=100)}]
    # rs = RandomizedSearchCV(estimator=pipeline, param_distributions=param_distributions,
    #                         cv=10, scoring='accuracy', refit=True, n_jobs=1,
    #                         random_state=1)
    #
    # rs.fit(X_train, y_train)

    # sklearn neural network
    # mlpc = MLPClassifier(learning_rate_init=0.07, activation='logistic', hidden_layer_sizes=5, max_iter=500)
    mlpc = MLPClassifier(hidden_layer_sizes=[8])
    mlpc.fit(X_train, y_train)

    return mlpc


def model_performance(mlpc, X_test, y_test):
    pred_mlpc = mlpc.predict(X_test)
    print(str(classification_report(y_test, pred_mlpc)) + "\n")
    print(str(confusion_matrix( y_test, pred_mlpc)) + "\n\n")

    fpr, tpr, thresholds = roc_curve(y_test[:-1], pred_mlpc[:-1])
    auc_mlp = auc(fpr, tpr)

    plt.figure(figsize=(5, 5), dpi=100)
    plt.plot(fpr, tpr, linestyle='-', label='MLP (auc = %0.3f)' % auc_mlp)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


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

    X_train, X_test, y_train, y_test = pre_processing(data_numeric)

    mlpc = mlp_model(X_train, y_train, X_test, y_test)

    model_performance(mlpc, X_test, y_test)

    # shap_usage(X_train, X_test, mlpc)


if __name__ == '__main__':
    main()
