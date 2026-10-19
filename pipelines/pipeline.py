"""SageMaker Pipeline (the CI/CD "DAG"): Preprocess -> Train -> Evaluate -> quality gate -> Register.

If the model misses the quality gate (ROC-AUC, recall, fairness gap), the pipeline ends in a
FailStep, which is the "failed" DAG state shown in the demo. Written for SageMaker Python SDK v2.
"""
import os
import shutil

from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo, ConditionLessThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

from src import config

FRAMEWORK_VERSION = "1.2-1"
INSTANCE_TYPE = "ml.m5.xlarge"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def stage_source(build_dir: str = os.path.join(REPO_ROOT, ".pipeline_build")) -> str:
    """Copy only the code SageMaker needs (src/ package + entry scripts) into a small folder."""
    shutil.rmtree(build_dir, ignore_errors=True)
    shutil.copytree(os.path.join(REPO_ROOT, "src"), os.path.join(build_dir, "src"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    for name in os.listdir(os.path.join(REPO_ROOT, "scripts")):
        if name.endswith(".py"):
            shutil.copy(os.path.join(REPO_ROOT, "scripts", name), build_dir)
    with open(os.path.join(build_dir, "requirements.txt"), "w") as f:
        f.write("pyarrow\n")            # parquet support inside the scikit-learn containers
    return build_dir


def get_pipeline(role: str, bucket: str, region: str = config.REGION,
                 name: str = config.PIPELINE_NAME, hyperparameters: dict = None) -> Pipeline:
    """`hyperparameters` are the tuned model settings (for example {"max-depth": 6})."""
    session = PipelineSession(default_bucket=bucket)
    source = stage_source()
    raw = f"s3://{bucket}/{config.RAW_PREFIX}"

    min_auc = ParameterFloat("MinRocAuc", default_value=config.MIN_ROC_AUC)
    min_recall = ParameterFloat("MinRecall", default_value=config.MIN_RECALL)
    max_gap = ParameterFloat("MaxFairnessGap", default_value=config.MAX_FAIRNESS_GAP)
    model_type = ParameterString("ModelType", default_value="gb")

    # 1. Preprocess -------------------------------------------------------------------
    processor = FrameworkProcessor(estimator_cls=SKLearn, framework_version=FRAMEWORK_VERSION, role=role,
                                   instance_type=INSTANCE_TYPE, instance_count=1, sagemaker_session=session)
    step_prep = ProcessingStep(
        name="Preprocess",
        step_args=processor.run(
            code="preprocess.py", source_dir=source,
            inputs=[ProcessingInput(f"{raw}/train/", "/opt/ml/processing/input/raw/train"),
                    ProcessingInput(f"{raw}/test/", "/opt/ml/processing/input/raw/test")],
            outputs=[ProcessingOutput(output_name=n, source=f"/opt/ml/processing/output/{n}")
                     for n in ("train", "test", "production")]))
    prep_out = step_prep.properties.ProcessingOutputConfig.Outputs

    # 2. Train ------------------------------------------------------------------------
    estimator = SKLearn(entry_point="train.py", source_dir=source, framework_version=FRAMEWORK_VERSION,
                        role=role, instance_type=INSTANCE_TYPE, instance_count=1, sagemaker_session=session,
                        output_path=f"s3://{bucket}/{config.MODEL_PREFIX}/pipeline",
                        hyperparameters={**(hyperparameters or {}), "model-type": model_type})
    step_train = TrainingStep(
        name="Train",
        step_args=estimator.fit(inputs={
            "train": TrainingInput(prep_out["train"].S3Output.S3Uri, content_type="application/x-parquet"),
            "test": TrainingInput(prep_out["test"].S3Output.S3Uri, content_type="application/x-parquet")}))

    # 3. Evaluate ---------------------------------------------------------------------
    report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    eval_processor = FrameworkProcessor(estimator_cls=SKLearn, framework_version=FRAMEWORK_VERSION, role=role,
                                        instance_type=INSTANCE_TYPE, instance_count=1, sagemaker_session=session)
    step_eval = ProcessingStep(
        name="Evaluate",
        step_args=eval_processor.run(
            code="evaluate.py", source_dir=source,
            inputs=[ProcessingInput(step_train.properties.ModelArtifacts.S3ModelArtifacts, "/opt/ml/processing/model"),
                    ProcessingInput(prep_out["test"].S3Output.S3Uri, "/opt/ml/processing/test")],
            outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")]),
        property_files=[report])

    # 4. Register (only reached when the gate passes) --------------------------------
    model = SKLearnModel(model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts, role=role,
                         entry_point="inference.py", source_dir=source,
                         framework_version=FRAMEWORK_VERSION, sagemaker_session=session)
    metrics_uri = Join(on="/", values=[step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                                       "evaluation.json"])
    step_register = ModelStep(
        name="RegisterChurnModel",
        step_args=model.register(
            content_types=["text/csv"], response_types=["text/csv"],
            inference_instances=["ml.m5.large"], transform_instances=["ml.m5.large"],
            model_package_group_name=config.MODEL_PACKAGE_GROUP, approval_status="PendingManualApproval",
            model_metrics=ModelMetrics(model_statistics=MetricsSource(s3_uri=metrics_uri,
                                                                     content_type="application/json"))))

    # 5. Quality gate -----------------------------------------------------------------
    def metric(path):
        return JsonGet(step_name=step_eval.name, property_file=report, json_path=path)

    step_fail = FailStep(name="QualityGateFailed",
                         error_message="Model did not meet the ROC-AUC, recall, or fairness thresholds")
    step_gate = ConditionStep(
        name="QualityGate",
        conditions=[ConditionGreaterThanOrEqualTo(left=metric("metrics.roc_auc.value"), right=min_auc),
                    ConditionGreaterThanOrEqualTo(left=metric("metrics.recall.value"), right=min_recall),
                    ConditionLessThanOrEqualTo(left=metric("fairness_max_gap.gender"), right=max_gap),
                    ConditionLessThanOrEqualTo(left=metric("fairness_max_gap.age_band"), right=max_gap)],
        if_steps=[step_register], else_steps=[step_fail])

    return Pipeline(name=name, parameters=[min_auc, min_recall, max_gap, model_type],
                    steps=[step_prep, step_train, step_eval, step_gate], sagemaker_session=session)
