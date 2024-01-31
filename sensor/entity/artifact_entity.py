from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_path:str
    train_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    drift_status:bool
    drift_report_path:str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str
    transformed_train_file_path:str
    transformed_test_file_path:str

@dataclass
class ClassificationMetricsArtifact:
    f1_score:float
    precision_score:float
    recall_score:float
    auroc_score:float
@dataclass
class ModelTrainerArtifact:
    trained_model_path:str
    train_metric_artifact:ClassificationMetricsArtifact
    test_metric_artifact:ClassificationMetricsArtifact
    
@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    improved_accuracy:float
    latest_model_path:str
    trained_model_path:str
    trained_model_metric_artifact:ClassificationMetricsArtifact
    latest_model_metric_artifact:ClassificationMetricsArtifact

@dataclass
class ModelPusherArtifact:
    saved_model_path:str
    model_file_path:str
