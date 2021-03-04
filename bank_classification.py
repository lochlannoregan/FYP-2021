# !/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score, plot_roc_curve
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import scipy as sc
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.compose import ColumnTransformer


def import_data():
    return pd.read_csv("data/bank/bank.csv", sep=";")


def pre_processing(data):
    numerical_features = data.select_dtypes(include='int64')
    categorical_features = data.select_dtypes(include='object')
    X = data.drop(['y'], axis=1)
    y = data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    print(Counter(y_train).items())
    X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)
    print(Counter(y_train_resampled).items())

    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # scaler.transform(X_train)
    # scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, numerical_features, categorical_features
    # return X_train_resampled, X_test, y_train_resampled, y_test


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

    # data_numeric = data[['age', 'job_encoded', 'marital_encoded', 'education_encoded', 'default_encoded', 'balance',
    #                      'housing_encoded', 'loan_encoded', 'contact_encoded', 'day', 'month_encoded',
    #                      'duration', 'campaign', 'pdays', 'previous', 'poutcome_encoded', 'y_encoded']].copy()

    data_numeric = data[
        ['age', 'default_encoded', 'balance', 'housing_encoded', 'loan_encoded', 'duration', 'campaign', 'pdays',
         'previous', 'poutcome_encoded', 'y_encoded']].copy()

    return data_numeric


def mlp_model(data):
    numerical_features = data.select_dtypes(include='int64').columns
    categorical_features = data.select_dtypes(include='object').drop(['y'], axis=1).columns
    X = data.drop(['y'], axis=1)
    y = data['y']

    numerical_transformer = Pipeline(steps=[('scalar', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)])

    # preprocessor.fit_transform(X, y)

    # smt = SMOTE(random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    # print(Counter(y_train).items())
    # X_train, y_train = SMOTE().fit_resample(X_train, y_train)

    steps = [('preprocessor', preprocessor),
             ('MLP', MLPClassifier(hidden_layer_sizes=[8, 8], activation='logistic'))]

    pipeline = Pipeline(steps)

    pipeline.fit(X_train, y_train)

    # parameters = {'MLP__hidden_layer_sizes': [8, 9, (8, 8), (9, 9)], 'MLP__activation': ['logistic', 'tanh', 'relu']}
    #
    # grid = GridSearchCV(pipeline, param_grid=parameters, scoring='roc_auc')
    #
    # grid.fit(X_train, y_train)
    # print(grid.best_params_)
    # # print(grid.best_score_)
    # print(grid.cv_results_)

    # mlpc = MLPClassifier(hidden_layer_sizes=[48, 20, 15], activation='logistic')
    # mlpc.fit(X_train, y_train)

    model_performance(pipeline, X_test, y_test)


def model_performance(mlpc, X_test, y_test):
    pred_mlpc = mlpc.predict(X_test)
    print(str(classification_report(y_test, pred_mlpc)) + "\n")
    print(str(confusion_matrix(y_test, pred_mlpc)) + "\n\n")

    plot_roc_curve(mlpc, X_test, y_test)
    plt.show()

    # fpr, tpr, thresholds = roc_curve(y_test[:-1], pred_mlpc[:-1])
    # auc_mlp = auc(fpr, tpr)
    #
    # plt.figure(figsize=(5, 5), dpi=100)
    # plt.plot(fpr, tpr, linestyle='-', label='MLP (auc = %0.3f)' % auc_mlp)
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.legend()
    # plt.show()


def shap_usage(X_train, X_test, mlpc):
    explainer = shap.KernelExplainer(mlpc.predict_proba, X_train)
    shap_values = explainer.shap_values(X_test.iloc[0, :])
    shap.force_plot(explainer.expected_value[0], shap_values[0], X_test.iloc[0, :], matplotlib=True)

    # shap_values = explainer.shap_values(X_test)
    # shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)


def main():
    shap.initjs()

    data = import_data()

    mlp_model(data)

    # shap_usage(X_train, X_test, mlpc)


if __name__ == '__main__':
    main()
