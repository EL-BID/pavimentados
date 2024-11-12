<div align="center">
  <h1>Pavimentados</h1>

[User Manual](docs/manual_202410.pdf)

![analytics image (flat)](https://raw.githubusercontent.com/vitr/google-analytics-beacon/master/static/badge-flat.gif)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=EL-BID_pavimentados&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=EL-BID_pavimentados)
[![Downloads](https://pepy.tech/badge/pavimentados)](https://pepy.tech/project/pavimentados)

![video_results.gif](docs/assets/video_results.gif)
</div>

## Description
---
Pavimentados is a tool that allows the identification of pavement faults located on highways or roads. 
This library provides an environment to use computer vision models developed to detect different elements. 
The detected elements are then used to generate metrics that aid the process of planning road maintenance.

The model files can be downloaded from this [link](https://github.com/EL-BID/pavimentados/raw/feature/v1.0.0/models/model_20240818.tar.gz?download=).

> **Important changes**: Unlike the previous version, this new version does not include traffic sign detection. We hope to be able to 
> include it again in future versions. 

## Main Features
---

Some of the features available are:

- Scoring using the models already developed.
- Workflows for image acquisition and assessment.
- Evaluation of gps information to determine the location of faults and elements.
- Image or video support.
- Support for GPS data in different formats (GPRRA, csv, embedded in image).

## Instalation
---

Install the library using the following command:

```
pip install pavimentados
```

Next, 
* [download the model](https://github.com/EL-BID/pavimentados/raw/feature/v1.0.0/models/model_20240818.tar.gz?download=) from this link

* Decompress it using the following command
```bash
tar -xzvf model_20240818.tar.gz
```

## Quick Start
---

In the `notebooks` folder there is a complete example of how to process both images and videos present
in `notebooks/road_videos` and `notebooks/road_images`. The results are saved to `notebooks/outputs`.

The first step is to import the components that create a workflow with images:
```
from pavimentados.processing.processors import MultiImage_Processor
from pavimentados.processing.workflows import Workflow_Processor
```

In this example, there is the image processor object MultiImage_Processor which is in charge of taking the images and analyzing them individually using the models. In addition, there is the Workflow_Processor object that is in charge of the image processing workflow. 

Internally, the Workflow_Processor has objects that can interpret different image sources or GPS information. 

Among the allowed image sources are:

 - image_routes: A list of image routes.
 - image_folder: A folder with all images.
 - video: The path to a video file.

Among the allowed GPS data sources are:

 - image_routes: A list of paths to the routes of the images that have the gps data embedded in them.
 - image_folder: A path to a folder with all the images that have the gps data embedded in them.
 - loc: A file in [NMEA format](docs%2Fgps_data_formats.md).
 
Once these elements are imported, the processor is instantiated as follows:

```python
from pathlib import Path
models_path = Path("./artifacts")  # Path to downloaded model
ml_processor = MultiImage_Processor(artifacts_path=str(models_path))
```

Alternatively, an additional JSON file can be specified to set or overwrite certain configuration parameters of the models.

```python
ml_processor = MultiImage_Processor(artifacts_path=str(models_path), config_file="./models_config.json")
```
These parameters allow the specification of parameter such as the confidence, iou, or maximum amount of detections per frame.

Example of the configuration file:
```json
{
	"paviment_model": {
		"yolo_threshold": 0.20,
		"yolo_iou": 0.45,
		"yolo_max_detections": 100
	}
}
```

The workflow object receives the instantiated processor. Without it is not able to execute the workflow.

```python
input_video_file = "sample.mp4"
input_gps_file = "sample.log"

# Create a workflow for videos
workflow = Workflow_Processor(
    input_video_file, image_source_type="video", gps_source_type="loc", gps_input=input_gps_file, adjust_gps=True
)
```

The last step is to execute the workflow:

```python
results = workflow.execute(ml_processor, 
                           batch_size=16, 
                           video_output_file="processed_video.mp4" 
                           )
```

>  * `video_output_file` and `image_folder_output` are optional and are only to save output video or image files along detections.

The results can be saved in csv format or used for further processing.

```python
# Save results to outputs directory
import pandas as pd
for result_name in results.keys():
    pd.DataFrame(results[result_name]).to_csv(f"{result_name}.csv")
```

In the `results` object you will find the following:

 1. table_summary_sections: DataFrame with summary table by sections.
 2. data_resulting: DataFrame with results per frame.
 3. data_resulting_fails: DataFrame with results by unique faults encountered.

To see more details about the results please refer to [this page](docs%2Fresults.md).


## Project structure
---
* `docs`: Documentation files.
* `models`: Reference path where the downloaded model artifact should be placed. 
* `notebooks`: Examples of how to process images and videos.
* `pavimentados/analyzers`: Modules for image/video processing and generation of the final output.
* `pavimentados/configs`: General configuration and parameters of the models.
* `pavimentados/models`: Modules for YoloV8 and Siamese models.
* `pavimentados/processing`: Workflows for processing.


## Changelog
---
For information regarding the latest changes/updates in the library please refer to the [changes document](docs/CHANGELOG.md).


## Authors
---

This package has been developed by:

<a href="https://github.com/J0s3M4rqu3z" target="blank">Jose Maria Marquez Blanco</a>
<br/>
<a href="https://www.linkedin.com/in/joancerretani/" target="blank">Joan Alberto Cerretani</a>
<br/>
<a href="https://www.linkedin.com/in/ingvictordurand/" target="blank">Victor Durand</a>
