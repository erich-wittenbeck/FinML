
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from lib.utility.statistics import avg

def roc_auc_multiclass_scorer(classes, average=None):

    def actual_scorer(model, X, y):

        estimation_function = model.estimation_func
        scores = estimation_function(X[model.features])

        result = 0.0

        if average == 'micro':
            if len(classes) > 2:
                y_bin = label_binarize(y, classes)
                fpr, tpr, _ = roc_curve(y_bin.ravel(), scores.ravel())
                result = auc(fpr, tpr)
            else:
                pos_class = classes[1]
                neg_class = classes[0]
                fpr, tpr, _ = roc_curve(y.apply(lambda val: pos_class if val == pos_class else neg_class), scores, pos_label=pos_class)
        elif average == 'macro':
            aucs = []
            for cls, cls_idx in zip(classes, range(len(classes))):
                if len(classes) > 2:
                    scores_cls = scores[:, cls_idx]
                else:
                    scores_cls = scores
                fpr, tpr, _ = roc_curve(y, scores_cls, pos_label=cls)
                aucs += [auc(fpr, tpr)]
            result = avg(aucs)
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