from pathlib import Path

import pandas as pd

from pavimentados.processing.processors import MultiImage_Processor
from pavimentados.processing.workflows import Workflow_Processor

if __name__ == "__main__":
    # Parameters
    input_path = Path("./road_videos")
    models_path = Path("../models/artifacts")

    input_video_file = input_path / "20230403M-F-P01.mp4"
    input_gps_file = input_path / "20230403M-F-P01.log"

    # Create processor
    ml_processor = MultiImage_Processor(artifacts_path=str(models_path))

    # Create workflow
    workflow = Workflow_Processor(
        input_video_file, image_source_type="video", gps_source_type="loc", gps_input=input_gps_file, adjust_gps=True
    )

    # Process inputs
    results = workflow.execute(ml_processor, video_output_file="outputs/processed_video.mp4", batch_size=16)

    # Save results to outputs directory
    for result_name in results.keys():
        pd.DataFrame(results[result_name]).to_csv(f"./outputs/{result_name}.csv")
