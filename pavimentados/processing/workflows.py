import logging
from pathlib import Path
from typing import Union

import pandas as pd

from pavimentados.analyzers.calculators import Results_Calculator as calculator
from pavimentados.analyzers.gps_sources import GPS_Data_Loader
from pavimentados.configs.utils import Config_Basic
from pavimentados.processing.processors import MultiImage_Processor
from pavimentados.processing.sources import Image_Source_Loader
from pavimentados.processing.utils import create_video_from_results

logger = logging.getLogger(__name__)
pavimentados_path = Path(__file__).parent.parent


class Workflow_Processor(Config_Basic):
    """Workflow processor for processing images or videos with GPS data."""

    def __init__(self, images_input, **kwargs):
        super().__init__()
        logger.info("Loading workflow config...")
        config_file_default = pavimentados_path / "configs" / "workflows_general.json"
        self.load_config(config_file_default, None)

        image_source_type = kwargs.get("image_source_type", "image_folder")
        gps_source_type = kwargs.get("gps_source_type", "image_folder")
        gps_in = kwargs.get("gps_input", images_input if gps_source_type == image_source_type else None)
        adjust_gps = kwargs.get("adjust_gps", False)
        gps_sections_distance = kwargs.get("gps_sections_distance", 100)

        self.img_obj = Image_Source_Loader(image_source_type, self.config, images_input)
        self.gps_data = GPS_Data_Loader(gps_source_type, gps_in, self.config, **kwargs)
        if adjust_gps:
            self.gps_data.adjust_gps_data(self.img_obj.get_len())
        self.gps_data.generate_gps_metrics(gps_sections_distance)
        self.executed = False

    def _execute_model(self, processor, batch_size=8, video_output_file=None, image_folder_output=None):
        """Execute the model on the input data.

        Args:
            processor: Processor to use for executing the model.
            batch_size: Batch size to use for processing images.
            video_output_file: Output file for the processed video.
            image_folder_output: Output folder for the processed images.

        Returns:
            None
        """
        self.results = processor.process_images_group(
            self.img_obj, batch_size=batch_size, video_output_file=video_output_file, image_folder_output=image_folder_output
        )
        self.executed = True

    def process_result(self, min_fotogram_distance: int = 5) -> None:
        """Process the results to generate the final output.

        Args:
            min_fotogram_distance: Minimum distance between frames to consider for processing.

        Returns:
            None
        """
        logger.info("Processing results...")
        self.table_summary_sections, self.data_resulting, self.data_resulting_fails = calculator.generate_paviment_results(
            self.config,
            self.results,
            self.img_obj,
            self.gps_data,
            columns_to_have=self.classes_names_yolo_paviment,
            min_fotogram_distance=min_fotogram_distance,
        )
        self.signals_summary = calculator.generate_final_results_signal(
            self.config, self.results, self.gps_data, classes_names_yolo_signal=self.classes_names_yolo_signal
        )

    def get_results(self) -> dict[str, any]:
        """Get the results of the workflow.

        Returns:
            dict: Dictionary containing the results of the workflow.
        """
        if not self.executed:
            raise ValueError("Workflow not yet executed, use execute method to store the results after executing models")
        return {
            "table_summary_sections": self.table_summary_sections,
            "data_resulting": self.data_resulting,
            "data_resulting_fails": self.data_resulting_fails,
            "signals_summary": self.signals_summary,
            # "raw_results": self.results,
        }

    def execute(
        self,
        processor: MultiImage_Processor,
        min_fotogram_distance: int = 5,
        batch_size: int = 8,
        return_results: bool = True,
        image_folder_output: str = None,
        video_output_file: str = None,
        video_from_results: bool = True,
        video_detections: str = "all"
    ) -> Union[None, dict[str, any]]:
        """Execute the workflow.

        Args:
            processor: Processor to use for executing the model.
            min_fotogram_distance: Minimum distance between frames to consider for processing.
            batch_size: Batch size to use for processing images.
            return_results: Whether to return the results of the workflow.
            video_output_file: Output file for the processed video.
            image_folder_output: Output folder for the processed images.
            video_from_results: Whether to create a video from the results of the workflow. If it is `false`,
                the video will be created with unprocessed detections which is useful to test the models.
            video_detections: Whether to draw detections on the images. Can be 'all', 'paviment', 'signal' or 'none'.

        Returns:
            None | dict[str, any]: None if return_results is False, otherwise a dictionary containing the results of the workflow.
        """
        logger.info("Executing workflow...")

        if video_output_file and image_folder_output:
            raise ValueError("Cannot use video_output_file and image_folder_output at the same time")

        video_output_results_file = ''
        if video_from_results and video_output_file:
            video_output_results_file = video_output_file
            video_output_file = None
            image_folder_output = None

        self.classes_names_yolo_paviment = processor.processor.yolov8_paviment_model.classes_names
        self.classes_names_yolo_signal = processor.processor.yolov8_signal_model.classes_names
        self._execute_model(processor, batch_size=batch_size, video_output_file=video_output_file, image_folder_output=image_folder_output)
        self.process_result(min_fotogram_distance=min_fotogram_distance)
        results = self.get_results()

        if video_from_results and video_output_results_file and results:
            logger.info("Creating video from results...")
            create_video_from_results(
                processor=self.img_obj,
                signals_detections=pd.DataFrame(results["signals_summary"]),
                fails_detections=pd.DataFrame(results["data_resulting"]),
                video_output_results_file=video_output_results_file,
                video_detections=video_detections
            )

        logger.info("Workflow executed successfully")

        if return_results:
            return results
