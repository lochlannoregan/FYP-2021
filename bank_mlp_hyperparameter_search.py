from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from bank_dataset_preprocessing import load_and_preprocess
from bank_mlp_model import train_model


def hyper_parameter_search(X_train, y_train):
    mlp_classifier = MLPClassifier()

    parameters = {
        'hidden_layer_sizes': [(100), (16), (21)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'lbfgs', 'adam'],
        'max_iter': [200, 300, 400]
    }

    grid = GridSearchCV(mlp_classifier, param_grid=parameters, scoring='roc_auc', cv=5)

    grid.fit(X_train, y_train)
    print(grid.best_params_)
    print(grid.best_score_)
    print(grid.cv_results_)


if __name__ == "__main__":
    X, y = load_and_preprocess()
    hyper_parameter_search(X, y)
