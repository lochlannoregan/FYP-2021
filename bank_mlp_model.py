import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, plot_roc_curve, plot_confusion_matrix
from sklearn.neural_network import MLPClassifier

from bank_dataset_preprocessing import load_and_preprocess


def train_model(X, y, X_train, X_test, y_train, y_test):
    mlp_classifier = MLPClassifier(hidden_layer_sizes=[21], solver='adam', activation='logistic', max_iter=200)
    mlp_classifier.fit(X_train, y_train)
    prediction_mlp = mlp_classifier.predict(X_test)

    print(str(classification_report(y_test, prediction_mlp)) + "\n")
    plot_confusion_matrix(mlp_classifier, X_test, y_test)
    plt.show()

    plot_roc_curve(mlp_classifier, X_test, y_test)
    plt.show()

    joblib.dump(mlp_classifier, "saved-model-computation/mlp_classifier.joblib")

    return mlp_classifier, X_test, X_train, y_test, y_train


def load_model():
    mlp_classifier = joblib.load("saved-model-computation/mlp_classifier.joblib")
    return mlp_classifier


def main():
    X, y, X_train, X_test, y_train, y_test = load_and_preprocess()
    train_model(X, y, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()