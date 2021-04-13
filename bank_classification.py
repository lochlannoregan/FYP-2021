# !/usr/bin/env python3

from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report, plot_roc_curve, plot_confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import set_config
set_config(display='diagram')

def import_data():
    return pd.read_csv("data/bank/bank.csv", sep=";")


def grid_search(pipeline, X_train, y_train):

    # Grid Search
    parameters = {'MLP__hidden_layer_sizes': [8, 9, (8, 8), (9, 9)], 'MLP__activation': ['logistic', 'tanh', 'relu']}

    grid = GridSearchCV(pipeline, param_grid=parameters, scoring='roc_auc')

    grid.fit(X_train, y_train)
    print(grid.best_params_)
    # print(grid.best_score_)
    print(grid.cv_results_)


def mlp_model(data):
    numerical_features = data.select_dtypes(include='int64').columns
    categorical_features = data.select_dtypes(include='object').drop(['y'], axis=1).columns

    X = data.drop(['y'], axis=1)
    y = data['y']

    numerical_transformer = Pipeline(steps=[('scalar', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])
    # categorical_transformer = Pipeline(steps=[('ordinal', OrdinalEncoder())])
    # target_variable_transformer = Pipeline(steps=[('label', LabelEncoder())])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features),
        # ('label', target_variable_transformer, target_variable)
    ])

    smt = SMOTE(random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33, random_state=0)

    steps = [('preprocessor', preprocessor),
             ('smote', smt),
             ('MLP', MLPClassifier(hidden_layer_sizes=297, random_state=0, alpha=0.0029797690517799937, learning_rate_init=0.06258315722416379,
                                   momentum=0.057971214902612256, solver='sgd', validation_fraction=0.1, warm_start=True, nesterovs_momentum=True,
                                   early_stopping=False, beta_1=0.9))
             # ('RandomForest', RandomForestClassifier())
             ]

    pipeline = Pipeline(steps)

    pipeline.fit(X_train, y_train)

    model_performance(pipeline, X_test, y_test)

    X_train_transformed = pipeline.named_steps['preprocessor'].fit_transform(X_train)
    X_test_transformed = pipeline.named_steps['preprocessor'].fit(X_train).transform(X_test)
    shap_usage(X_train_transformed, X_test_transformed, pipeline)


def model_performance(pipeline, X_test, y_test):
    pred_mlpc = pipeline.predict(X_test)
    print(str(classification_report(y_test, pred_mlpc)) + "\n")
    # print(str(confusion_matrix(y_test, pred_mlpc)) + "\n")
    plot_confusion_matrix(pipeline, X_test, y_test)
    plt.show()

    plot_roc_curve(pipeline, X_test, y_test)
    plt.show()

    # Cross validation?
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # # evaluate model
    # scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    #
    # print('Mean ROC AUC: %.3f' % mean(scores))


def shap_usage(X_train, X_test, mlpc):
    shap.initjs()
    explainer = shap.KernelExplainer(mlpc.named_steps['MLP'].predict_proba, X_train)
    shap_values = explainer.shap_values(X_test[0])
    shap.force_plot(explainer.expected_value[0], shap_values[0], X_test[0], matplotlib=True)


def main():
    data = import_data()

    mlp_model(data)


if __name__ == '__main__':
    main()
