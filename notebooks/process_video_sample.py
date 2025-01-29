import logging
from pathlib import Path

import pandas as pd

from pavimentados.configs.utils import setup_logging
from pavimentados.processing.processors import MultiImage_Processor
from pavimentados.processing.workflows import Workflow_Processor

if __name__ == "__main__":
    # Parameters
    input_path = Path("./road_videos")
    models_path = Path("../models/artifacts")
    input_video_name = "sample"

    output_path = Path("./outputs") / input_video_name
    output_path.mkdir(parents=True, exist_ok=True)

    input_video_file = input_path / f"{input_video_name}.mp4"
    input_gps_file = input_path / f"{input_video_name}.log"

    # Setup logging
    setup_logging(level=logging.INFO)

    # Create processor
    ml_processor = MultiImage_Processor(artifacts_path=str(models_path), config_file="./models_config.json")

    # Create workflow
    workflow = Workflow_Processor(
        input_video_file, image_source_type="video", gps_source_type="loc", gps_input=input_gps_file, adjust_gps=True
    )

    # Process inputs
    results = workflow.execute(ml_processor, video_output_file=f"{output_path}/processed_video.mp4")

    # Save results to outputs directory
    for result_name in results.keys():
        pd.DataFrame(results[result_name]).to_csv(f"{output_path}/{result_name}.csv")
