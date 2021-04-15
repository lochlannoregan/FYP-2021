def learning_curve_calculation(mlp_classifier, X, y):
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    title = "Learning Curves (Naive Bayes)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = mlp_classifier
    learning_curve(estimator, title, X, y, cv=cv)

    title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = SVC(gamma=0.001)
    learning_curve(estimator, title, X, y, cv=cv)

    plt.show()







    # scores = cross_validate(mlp_classifier, X_test, y_test, scoring=['roc_auc'], cv=5)
