import os
import sys
import time
import argparse
from urllib.parse import urlparse
import pathlib

import boto3


class DeployEndpoint:

    def __init__(self, args):
        self.arn_path = args.arn_path
        self.comprehend = boto3.client("comprehend", region_name=os.environ["AWS_REGION"])
        self.s3 = boto3.client("s3")

    def create_endpoint(self) -> str:
        """
        :return: endpoint arn
        """
        object_arn = (
            self.s3.get_object(
                Bucket=urlparse(self.arn_path).netloc,
                Key=urlparse(f"{self.arn_path}/arn.txt").path[1:],
            )["Body"].read().decode().strip())

        endpoint_response = self.comprehend.create_endpoint(
            EndpointName=f'DEMO-classifier-{time.strftime("%Y-%m-%d-%H-%M-%S")}',
            ModelArn=object_arn,
            DesiredInferenceUnits=10,
        )
        return endpoint_response["EndpointArn"]

    def get_endpoint_status(self, endpoint_arn: str) -> str:
        """
        :param endpoint_arn: arn of the new endpoint
        :return: endpoint status
        """
        describe_endpoint = self.comprehend.describe_endpoint(EndpointArn=endpoint_arn)
        return describe_endpoint["EndpointProperties"]["Status"]

    def deploy_endpoint(self):
        """
        :rtype: object
        """
        endpoint_arn = self.create_endpoint()

        max_time = time.time() + 15 * 60  # 15 min ??
        while time.time() < max_time:
            status = self.get_endpoint_status(endpoint_arn)
            if status == "IN_ERROR":
                sys.exit(1)

            if status == "IN_SERVICE":
                endpoint_arn_output_dir = "/opt/ml/processing/endpoint_arn"
                pathlib.Path(endpoint_arn_output_dir).mkdir(parents=True, exist_ok=True)

                print(f"Writing out endpoint arn {endpoint_arn}")
                endpoint_arn_path = f"{endpoint_arn_output_dir}/endpoint_arn.txt"
                with open(endpoint_arn_path, "w") as f:
                    f.write(endpoint_arn)
                break

            time.sleep(120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arn-path", type=str, help="Path to the Arn on S3")
    args = parser.parse_args()
    DeployEndpoint(args).deploy_endpoint()
