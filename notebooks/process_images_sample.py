import logging
from pathlib import Path

import pandas as pd

from pavimentados.configs.utils import setup_logging
from pavimentados.processing.processors import MultiImage_Processor
from pavimentados.processing.workflows import Workflow_Processor

if __name__ == "__main__":
    # Parameters
    input_path = Path("road_images")
    models_path = Path("../models/artifacts")

    output_path = Path("./outputs/road_images")
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(level=logging.INFO)

    # Create processor
    ml_processor = MultiImage_Processor(artifacts_path=str(models_path), config_file="./models_config.json")

    # Create workflow
    workflow = Workflow_Processor(input_path, image_source_type="image_folder", gps_source_type="image_folder")

    # Process inputs
    results = workflow.execute(ml_processor, image_folder_output=str(output_path))

    # Save results to outputs directory
    for result_name in results.keys():
        pd.DataFrame(results[result_name]).to_csv(f"{output_path}/{result_name}.csv")
