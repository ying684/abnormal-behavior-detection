#  core/training/evaluation.py

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)


class Evaluator:

    @staticmethod
    def evaluate(y_true, y_pred):

        acc = accuracy_score(
            y_true,
            y_pred
        )

        precision, recall, f1, _ = \
            precision_recall_fscore_support(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm
        }