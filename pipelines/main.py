import logging

import fire
from utils.actions import upsert_pipeline, run_pipeline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire({
        "upsert": upsert_pipeline,
        "run": run_pipeline
    })
