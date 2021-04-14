import joblib
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, plot_roc_curve, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from bank_dataset_preprocessing import load_and_preprocess


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33, random_state=42)

    mlp_classifier = MLPClassifier(hidden_layer_sizes=[21], solver='adam', activation='logistic', max_iter=200)
    mlp_classifier.fit(X_train, y_train)
    prediction_mlp = mlp_classifier.predict(X_test)

    print(str(classification_report(y_test, prediction_mlp)) + "\n")
    plot_confusion_matrix(mlp_classifier, X_test, y_test)
    plt.show()

    plot_roc_curve(mlp_classifier, X_test, y_test)
    plt.show()

    joblib.dump(mlp_classifier, "saved-model-computation/mlp_classifier.joblib")
    joblib.dump(X_test, "saved-model-computation/x_test.joblib")
    joblib.dump(X_train, "saved-model-computation/X_train.joblib")
    joblib.dump(y_test, "saved-model-computation/y_test.joblib")
    joblib.dump(y_train, "saved-model-computation/y_train.joblib")

    return mlp_classifier, X_test, X_train, y_test, y_train
    # joblib.dump(mlp_classifier, "saved-model-computation/mlp_classifier.joblib")
    # joblib.dump(X_test, "saved-model-computation/x_test.joblib")
    # joblib.dump(X_train, "saved-model-computation/X_train.joblib")
    # joblib.dump(y_test, "saved-model-computation/y_test.joblib")
    # joblib.dump(y_train, "saved-model-computation/y_train.joblib")
    #
    # return mlp_classifier, X_test, X_train, y_test, y_train


def load_model():
    mlp_classifier = joblib.load("saved-model-computation/mlp_classifier.joblib")
    X_test = joblib.load("saved-model-computation/x_test.joblib")
    X_train = joblib.load("saved-model-computation/X_train.joblib")
    y_test = joblib.load("saved-model-computation/y_test.joblib")
    y_train = joblib.load("saved-model-computation/y_train.joblib")

    return mlp_classifier, X_test, X_train, y_test, y_train
    # mlp_classifier = joblib.load("saved-model-computation/mlp_classifier.joblib")
    # X_test = joblib.load("saved-model-computation/x_test.joblib")
    # X_train = joblib.load("saved-model-computation/X_train.joblib")
    # y_test = joblib.load("saved-model-computation/y_test.joblib")
    # y_train = joblib.load("saved-model-computation/y_train.joblib")
    #
    # return mlp_classifier, X_test, X_train, y_test, y_train


def main():
    X, y = load_and_preprocess()
    train_model(X, y)


if __name__ == "__main__":
    main()