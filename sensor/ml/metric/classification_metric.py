import numpy as np
from sensor.logger import logging
from sensor.exceptions import SensorException
from sensor.entity.artifact_entity import ClassificationMetricsArtifact
from sklearn.metrics import (f1_score, recall_score, roc_auc_score, precision_score)

class ClassificationMetrics:
    @staticmethod
    def get_classfication_metric(y_true:np.array, y_pred:np.array)->ClassificationMetricsArtifact:
        f1score = f1_score(y_true , y_pred)
        recallscore = recall_score(y_true , y_pred)
        precisionscore = precision_score(y_true , y_pred)
        aurocscore = roc_auc_score(y_true , y_pred)

        classfication_metric_artifact = ClassificationMetricsArtifact(
            f1_score=f1score, precision_score=precisionscore
            , recall_score=recallscore, auroc_score=aurocscore
        )
        
        return classfication_metric_artifact
    


