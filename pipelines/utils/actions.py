from sagemaker.workflow.pipeline import Pipeline


def upsert_pipeline(pipeline: Pipeline, role_arn: str):
    pipeline.upsert(role_arn=role_arn)


def run_pipeline(pipeline: Pipeline):
    pipeline.start()
