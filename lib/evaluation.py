from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from lib.utility.ml import roc_auc_multiclass_scorer
from lib.utility.visualization import *

class Evaluator():

    def __init__(self):
        self.__models = {}
        self.__classification_reports = {}
        self.__confusion_matrices = {}
        self.__roc_curves = {}

    # Getters

    def __get_models(self):
        return self.__models

    def __get_reports(self):
        return self.__classification_reports

    def __get_confmats(self):
        return self.__confusion_matrices

    def __get_roc_curves(self):
        return self.__roc_curves

    # Private utility methods

    def __create_classification_reports(self, model, X, y):

        report = classification_report(y, y_pred=model.predict(X), output_dict=True)

        report.pop('accuracy', None) # Prevent bug later down the line

        self.__classification_reports[model.name] = report

    def __create_confusion_matrices(self, model, X, y):

        confmat = confusion_matrix(y_true=y, y_pred=model.predict(X))

        self.__confusion_matrices[model.name] = confmat

    def __create_roc_curves(self, model, X, y):
        classes = model.classes
        features = model.features
        scores = model.estimation_func(X[features])

        roc_dict = {}

        for cls, cls_idx in zip(classes, range(len(classes))):
            scores_cls = scores[:, cls_idx]

            fpr, tpr, threshold = roc_curve(y, scores_cls, pos_label=cls)
            roc_auc = auc(fpr, tpr)

            roc_dict[str(cls)] = {'fpr' : fpr,
                                  'tpr' : tpr,
                                  'threshold' : threshold,
                                  'roc_auc' : roc_auc}
        auc_macro = roc_auc_multiclass_scorer(classes, average='macro')
        auc_micro = roc_auc_multiclass_scorer(classes, average='micro')

        roc_dict['auc_macro'] = auc_macro(model, X, y)
        roc_dict['auc_micro'] = auc_micro(model, X, y)

        self.__roc_curves[model.name] = roc_dict

    # Public methods

    def evaluate(self, test_data, *models, evaluations=['reports', 'confmats', 'roc_curves']):

        X, y = test_data.X, test_data.y

        for model in models:
            self.__models[model.name] = model

            for evaluation in evaluations:
                if evaluation == 'reports':
                    self.__create_classification_reports(model, X, y)
                elif evaluation == 'confmats':
                    self.__create_confusion_matrices(model, X, y)
                elif evaluation == 'roc_curves':
                    self.__create_roc_curves(model, X, y)
                else:
                    raise ValueError(evaluation + ': Unknown evaluation type!')

        return self

    def plot(self, save_as=None, *args, **kwargs):
        nrows = sum([1 for field in [self.__classification_reports,
                                     self.__confusion_matrices,
                                     self.__roc_curves]
                     if field])
        ncols = len(self.__models)

        row = -1

        # TODO: Needs to be made more elegant

        if self.__classification_reports:
            names, reports = self.__classification_reports.keys(), self.__classification_reports.values()
            row += 1
            for col, name, report in zip(range(ncols), names, reports):
                ax = plt.subplot(nrows, ncols, row*ncols + col + 1)
                plot_classification_report(ax, report)

                if row == 0:
                    ax.set_title(name)
        if self.__confusion_matrices:
            names, confmats = self.__confusion_matrices.keys(), self.__confusion_matrices.values()
            row += 1
            for col, name, confmat in zip(range(ncols), names, confmats):
                ax = plt.subplot(nrows, ncols, row*ncols + col + 1)
                plot_confustion_matrix(ax, confmat, self.__models[name].classes)

                if row == 0:
                    ax.set_title(name)
        if self.__roc_curves:
            names, roc_dicts = self.__roc_curves.keys(), self.__roc_curves.values()
            row += 1
            for col, name, roc_dict in zip(range(ncols), names, roc_dicts):
                ax = plt.subplot(nrows, ncols, row*ncols + col + 1)
                plot_roc_curves(ax, roc_dict)

                if row == 0:
                    ax.set_title(name)

        plt.tight_layout()

        if save_as is None:
            plt.show()
        else:
            plt.savefig(save_as, *args, **kwargs)

        return self

    """ Properties """

    models = property(__get_models)
    reports = property(__get_reports)
    confmats = property(__get_confmats)
    roc_curves = property(__get_roc_curves)