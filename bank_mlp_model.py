import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, plot_roc_curve, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
import joblib
from bank_dataset_preprocessing import load_and_preprocess
from sklearn.linear_model import LogisticRegression


def train_model(X_test, X_train, y_test, y_train):
    # mlp_classifier = MLPClassifier(hidden_layer_sizes=297, random_state=0, alpha=0.0029797690517799937,
    #                                learning_rate_init=0.06258315722416379,
    #                                momentum=0.057971214902612256, solver='sgd', validation_fraction=0.1,
    #                                warm_start=True, nesterovs_momentum=True,
    #                                early_stopping=False, beta_1=0.9)
    # mlp_classifier = LogisticRegression()
    mlp_classifier = MLPClassifier(hidden_layer_sizes=[21])
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
    X, y = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33, random_state=42)

    mlp_classifier = train_model(X_test, X_train, y_test, y_train)
    # mlp_classifier, X_test, X_train, y_test, y_train = load_model()

    # shap(mlp_classifier, X_test, X_train, y_test, y_train)


if __name__ == "__main__":
    main()