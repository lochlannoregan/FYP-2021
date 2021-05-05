import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from bank_dataset_preprocessing import load_and_preprocess


def compare_models():

    X, y, X_train, X_test, y_train, y_test = load_and_preprocess()

    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('MLP', MLPClassifier()))
    models.append(('SVC', SVC()))
    models.append(('RF', RandomForestClassifier()))
    # evaluate each model in turn
    results = []
    names = []
    scoring_metric_selected='roc_auc'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=5)
        cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring_metric_selected)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle("Algorithm Comparison with " + str(scoring_metric_selected) + ' metric')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


if __name__ == '__main__':
    compare_models()
