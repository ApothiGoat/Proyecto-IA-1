from collections import defaultdict


def accuracy(y_true, y_pred):
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def precision_recall_f1(y_true, y_pred):
    classes = set(y_true)

    precision = {}
    recall = {}
    f1 = {}

    for c in classes:
        tp = sum((yt == c and yp == c) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != c and yp == c) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == c and yp != c) for yt, yp in zip(y_true, y_pred))

        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision[c] + recall[c] > 0:
            f1[c] = 2 * (precision[c] * recall[c]) / (precision[c] + recall[c])
        else:
            f1[c] = 0

    return precision, recall, f1


def macro_f1(f1_scores):
    return sum(f1_scores.values()) / len(f1_scores)


def confusion_matrix(y_true, y_pred):
    classes = sorted(list(set(y_true)))
    matrix = {c: {c2: 0 for c2 in classes} for c in classes}

    for yt, yp in zip(y_true, y_pred):
        matrix[yt][yp] += 1

    return matrix  