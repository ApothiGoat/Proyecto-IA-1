import random


def k_fold_split(X, y, k=5):
    """
    Divide los datos en K folds.
    """

    data = list(zip(X, y))
    random.shuffle(data)

    fold_size = len(data) // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size

        test = data[start:end]
        train = data[:start] + data[end:]

        X_train, y_train = zip(*train)
        X_test, y_test = zip(*test)

        folds.append((list(X_train), list(y_train), list(X_test), list(y_test)))

    return folds