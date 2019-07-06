
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from lib.utility.statistics import avg

def roc_auc_multiclass_scorer(classes, average=None):

    def actual_scorer(estimator, X, y):

        decision_function = estimator.decision_function if hasattr(estimator, 'decision_function') else estimator.predict_proba
        scores = decision_function(X)

        result = 0.0

        if average == 'micro':
            y_bin = label_binarize(y, classes)
            fpr, tpr, _ = roc_curve(y_bin.ravel(), scores.ravel())
            result = auc(fpr, tpr)
        elif average == 'macro':
            aucs = []
            for cls, cls_idx in zip(classes, range(len(classes))):
                fpr, tpr, _ = roc_curve(y, scores[:, cls_idx], pos_label=cls)
                aucs += [auc(fpr, tpr)]
            result = avg(*aucs)
        elif average == 'weighted':
            support = count_class_occurences(y, classes)
            aucs = []
            for cls, cls_idx in zip(classes, range(len(classes))):
                fpr, tpr, _ = roc_curve(y, scores[:, cls_idx], pos_label=cls)
                aucs += [auc(fpr, tpr)]*support[cls_idx]
            result = avg(aucs)
        else:
            pass

        return result

    return actual_scorer

def count_class_occurences(data, classes):

    return [len([d for d in data if d == cls]) for cls in classes]