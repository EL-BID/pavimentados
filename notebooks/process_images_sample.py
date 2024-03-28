from pathlib import Path

import pandas as pd
import tensorflow as tf

from pavimentados.processing.processors import MultiImage_Processor
from pavimentados.processing.workflows import Workflow_Processor

if __name__ == "__main__":
    GPU = False
    input_path = Path("road_images/")
    models_path = Path("../models/artifacts")

    # Create processor
    ml_processor = MultiImage_Processor(assign_devices=True, gpu_enabled=GPU, artifacts_path=str(models_path))

    # Create workflow
    workflow = Workflow_Processor(input_path, image_source_type="image_folder", gps_source_type="image_folder")

    # Process inputs
    results = workflow.execute(ml_processor)

    # Save results to outputs directory
    for result_name in results.keys():
        pd.DataFrame(results[result_name]).to_csv(f"./outputs/{result_name}.csv")
