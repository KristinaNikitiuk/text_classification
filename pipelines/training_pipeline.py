import boto3
from sagemaker.sklearn.processing import SKLearnProcessor

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.parameters import ParameterInteger, ParameterString

from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

from sagemaker.workflow.pipeline import Pipeline

sagemaker_session = boto3.session.Session()
region = sagemaker_session.region_name
role_arn = 'arn:aws:iam::823254927476:role/service-role/AmazonSageMaker-ExecutionRole-20231102T173201'
processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
model_output = ParameterString(name="ModelOutput", default_value=f"s3://businesssolver-test-data/test/model")
input_train = ParameterString(
    name="TrainData",
    default_value="s3://businesssolver-test-data/test/df_train.csv",
)

input_test = ParameterString(
    name="TestData",
    default_value="s3://businesssolver-test-data/test/df_test.csv",
)


def get_pipeline():
    sklearn_processor = SKLearnProcessor(
        framework_version="0.20.0",
        role=role_arn,
        instance_type="ml.m5.xlarge",
        instance_count=1,
    )

    # PREPROCESS
    preprocess = ProcessingStep(
        name="ComprehendProcess",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(source=input_train, destination="/opt/ml/processing/input_train"),
            ProcessingInput(source=input_test, destination="/opt/ml/processing/input_test"),
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code="preprocessing/prepare_data.py",
    )

    # TRAIN
    evaluation_report = PropertyFile(
        name="ComprehendEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    comprehend_train_and_eval = ProcessingStep(
        name="ComprehendTrainAndEval",
        processor=sklearn_processor,
        job_arguments=[
            "--train-input-file",
            preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            "--train-output-path",
            model_output,
            "--iam-role-arn",
            role_arn,
        ],
        code="training/train_comprehend_classifier.py",
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
            ProcessingOutput(output_name="arn", source="/opt/ml/processing/arn"),
        ],
        property_files=[evaluation_report],
    )

    # DEPLOY ENDPOINT
    step_deploy_model = ProcessingStep(
        name="ComprehendDeploy",
        processor=sklearn_processor,
        job_arguments=[
            "--arn-path",
            comprehend_train_and_eval.properties.ProcessingOutputConfig.Outputs["arn"].S3Output.S3Uri,
        ],
        code="utils/deploy_endpoint.py",
        outputs=[
            ProcessingOutput(output_name="endpoint_arn", source="/opt/ml/processing/endpoint_arn")
        ],
    )

    cond_lte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name="ComprehendTrainAndEval",
            property_file=evaluation_report,
            json_path="Accuracy",
        ),
        right=0.65,
    )

    step_cond = ConditionStep(
        name="ComprehendAccuracyCondition",
        conditions=[cond_lte],
        if_steps=[step_deploy_model],
        else_steps=[],
    )

    pipeline_name = "DEMO-ComprehendPipeline"
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            input_train,
            input_test,
            model_output,
        ],
        steps=[preprocess, comprehend_train_and_eval, step_cond],
    )

    return pipeline
