from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report, plot_roc_curve, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn import set_config
from sklearn.impute import SimpleImputer
import pickle
import joblib


def train_save_model():
    data = pd.read_csv("data/bank-additional/bank-additional.csv", sep=";")

    # numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    # categorical_features = data.select_dtypes(include='object').drop(['y'], axis=1).columns

    numerical_features = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                          'euribor3m', 'nr.employed', 'duration']
    categorical_features_nominal = ['job', 'marital', 'contact', 'month', 'day_of_week']
    categorical_features_binary = ['default', 'housing', 'loan']
    categorical_features_unknown_values = ['education', 'default', 'housing', 'loan', 'job', 'marital']

    X = data.drop(['y', 'poutcome'], axis=1)
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

    # One hot encoding categorical nominal variables
    X_dummies = pd.get_dummies(X[categorical_features_nominal], drop_first=True)
    # Dropping
    X = X.drop(categorical_features_nominal, axis='columns')
    # Concatenating
    X = pd.concat([X, X_dummies], axis='columns')

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33, random_state=42)

    # print(y_train.describe())
    #
    # X_train, y_train = SMOTE(random_state=0).fit_resample(X_train, y_train)
    #
    # print(y_train.describe())

    # # Compare Algorithms
    # import pandas
    # import matplotlib.pyplot as plt
    # from sklearn import model_selection
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.neighbors import KNeighborsClassifier
    # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    # from sklearn.naive_bayes import GaussianNB
    # from sklearn.svm import SVC
    #
    # models = []
    # models.append(('LR', LogisticRegression()))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    # models.append(('SVM', SVC()))
    # models.append(('MLP', MLPClassifier()))
    # # evaluate each model in turn
    # results = []
    # names = []
    # scoring = 'accuracy'
    # for name, model in models:
    #     kfold = model_selection.KFold(n_splits=10)
    #     cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    #     results.append(cv_results)
    #     names.append(name)
    #     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #     print(msg)
    # # boxplot algorithm comparison
    # fig = plt.figure()
    # fig.suptitle('Algorithm Comparison')
    # ax = fig.add_subplot(111)
    # plt.boxplot(results)
    # ax.set_xticklabels(names)
    # plt.show()

    mlp_classifier = MLPClassifier(hidden_layer_sizes=297, random_state=0, alpha=0.0029797690517799937,
                                   learning_rate_init=0.06258315722416379,
                                   momentum=0.057971214902612256, solver='sgd', validation_fraction=0.1,
                                   warm_start=True, nesterovs_momentum=True,
                                   early_stopping=False, beta_1=0.9)
    mlp_classifier.fit(X_train, y_train)
    prediction_mlp = mlp_classifier.predict(X_test)

    print(str(classification_report(y_test, prediction_mlp)) + "\n")
    plot_confusion_matrix(mlp_classifier, X_test, y_test)
    plt.show()

    plot_roc_curve(mlp_classifier, X_test, y_test)
    plt.show()

    joblib.dump(mlp_classifier, "mlp_classifier.joblib")
    joblib.dump(X_test, "x_test.joblib")
    joblib.dump(X_train, "X_train.joblib")
    joblib.dump(y_test, "y_test.joblib")
    joblib.dump(y_train, "y_train.joblib")

    return mlp_classifier, X_test, X_train, y_test, y_train


def load_model():
    mlp_classifier = joblib.load("mlp_classifier.joblib")
    X_test = joblib.load("x_test.joblib")
    X_train = joblib.load("X_train.joblib")
    y_test = joblib.load("y_test.joblib")
    y_train = joblib.load("y_train.joblib")

    return mlp_classifier, X_test, X_train, y_test, y_train


def shap(mlp_classifier, X_test, X_train, y_test, y_train):
    #
    # import shap
    # shap.initjs()
    # explainer = shap.KernelExplainer(mlp_classifier.predict_proba, X_train)
    # shap_values = explainer.shap_values(X_test.iloc[0], nsamples=1000)
    # shap.force_plot(explainer.expected_value[0], shap_values[0], X_test.iloc[0], matplotlib=True)

    # import shap
    # shap.initjs()
    # f = lambda x: mlp_classifier.predict_proba(x)[:, 1]
    # med = X_train.median().values.reshape((1, X_train.shape[1]))
    # explainer = shap.Explainer(f, med)
    # shap_values = explainer(X_test.iloc[0:1000, :])
    # shap.plots.waterfall(shap_values[0])

    # import shap
    # shap.initjs()
    # X_train_summary = shap.kmeans(X_train, 10)
    # explainer = shap.KernelExplainer(mlp_classifier.predict, X_train_summary)
    # shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values, X_test)

    import shap
    shap.initjs()
    X_train_summary = shap.kmeans(X_train, 10)
    explainer = shap.KernelExplainer(mlp_classifier.predict, X_train_summary)
    shap_values = explainer.shap_values(X_test)
    shap.dependence_plot("duration", shap_values, X_test)


def main():
    # mlp_classifier, X_test, X_train, y_test, y_train = train_save_model()
    mlp_classifier, X_test, X_train, y_test, y_train = load_model()
    shap(mlp_classifier, X_test, X_train, y_test, y_train)


if __name__ == "__main__":
    main()