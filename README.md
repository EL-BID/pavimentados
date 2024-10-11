![analytics image (flat)](https://raw.githubusercontent.com/vitr/google-analytics-beacon/master/static/badge-flat.gif)
![analytics](https://www.google-analytics.com/collect?v=1&cid=555&t=pageview&ec=repo&ea=open&dp=/pavimentados/readme&dt=&tid=UA-4677001-16)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=EL-BID_pavimentados&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=EL-BID_pavimentados)
[![Downloads](https://pepy.tech/badge/pavimentados)](https://pepy.tech/project/pavimentados)
# Pavimentados

![video_results.gif](docs/assets/video_results.gif)

## Description
---
Pavimentados is a tool that allows the identification of pavement faults located on highways or roads, as well as vertical and horizontal signage. 
This library is an environment around the computer vision models developed to allow the detection of the different elements. 
The faults and detected elements are then used to generate metrics that help in road maintenance planning.

You can download models files from this [link](https://github.com/EL-BID/pavimentados/raw/feature/v1.0.0/models/model_20240818.tar.gz?download=).

These models require images or videos taken with the specifications that will be explained in later sections. 

So far the system uses 3 models involved in different phases of detection and categorization.

| Model Name             | Description                                         | Classes |
|------------------------|---------------------------------------------------- |---------|
| Paviment failures      | Detection of failures on the road and classifies it | 8       |
| Signage detection      | Detection of signage on the road                    | 1       |
| Signage classification | Classifies the signage detected                     | 314     |

To understand each model in detail see the [models](docs/MODELS.md) section.

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

To install you can use the following commands:

```
pip install pavimentados
```

The next step is to download the model artifact and decompress it.

```
wget -O models.tar.gz https://github.com/EL-BID/pavimentados/raw/feature/v1.0.0/models/model_20240818.tar.gz?download=
tar -xzvf models.tar.gz
```

## Quick Start
---

In the `notebooks` folder you find a complete example of how to process both images and videos present
in `notebooks/road_videos` or `notebooks/road_images` and save the results in `notebooks/outputs`.

The first step is to import the components that create a workflow with images:
```
from pavimentados.processing.processors import MultiImage_Processor
from pavimentados.processing.workflows import Workflow_Processor
```

In this example, we have the image processor object MultiImage_Processor, which is in charge of taking the images and analyzing them individually using the models. In addition, we have the Workflow_Processor object that 
is in charge of the image processing workflow. 

Internally, the Workflow_Processor has objects that can interpret different image sources or GPS information. 

Among the allowed image sources we have:

 - image_routes: A list of image routes
 - image_folder: A folder with all images
 - images: Images already loaded in numpy format
 - video: The path to a video file

Among the allowed GPS data sources we have:

 - image_routes: A list of the routes of the images that have the gps data embedded in them.
 - image_folder: A folder with all the images that have the gps data embedded in them.
 - loc: A file in GPRRA format
 - csv: A gps file with the gps information in columns and rows.

Once these elements are imported, the processor is instantiated as follows:

```python
from pathlib import Path
models_path = Path("./artifacts")  # Path to downloaded model
ml_processor = MultiImage_Processor(artifacts_path=str(models_path))
```

Alternatively, you can specify an additional JSON file for the setting or overwrite some of certain configuration 
parameters on the models.

```
ml_processor = MultiImage_Processor(artifacts_path=str(models_path), config_file="./models_config.json")
```
These parameters allow to specify for example the confidence, iou or maximum amount of detections per frame.

Example of configuration file:
```json
{
	"signal_model": {
		"yolo_threshold": 0.20,
		"yolo_iou": 0.45,
		"yolo_max_detections": 100
	},

	"paviment_model": {
		"yolo_threshold": 0.20,
		"yolo_iou": 0.45,
		"yolo_max_detections": 100
	}
}
```

The workflow object is able to receive the instantiated processor, without it is not able to execute the workflow.

```
input_video_file = "sample.mp4"
input_gps_file = "sample.log"

# Create a workflow for videos
workflow = Workflow_Processor(
    input_video_file, image_source_type="video", gps_source_type="loc", gps_input=input_gps_file, adjust_gps=True
)
```

The last step is to execute the workflow:

```
results = workflow.execute(ml_processor, batch_size=16, 
                           video_output_file="processed_video.mp4", 
                           video_from_results=True,
                           video_detections="all"
                           )
```

>  * `video_output_file` and `image_folder_output` are optional and are only to save output video or image 
files along detections.
>  * `video_from_results=True` is only to create a video from the results of the workflow. If it is `false`, the video will be created with unprocessed detections which is useful to test the models.
>  * `video_detections="all"` draws detections on the images. Can be 'all', 'only_fails' or 'only_signals'.

The next step is to save the results in csv format or use it for further processing.

```
# Save results to outputs directory
import pandas as pd
for result_name in results.keys():
    pd.DataFrame(results[result_name]).to_csv(f"{result_name}.csv")
```

In the `results` object you will find the following:

 1. table_summary_sections: DataFrame with summary table by sections.
 2. data_resulting: DataFrame with results per frame.
 3. data_resulting_fails: DataFrame with results by unique faults encountered.
 4. signals_summary: DataFrame with all the information about the signals.

To see more detail about results please refer to [this page.](docs%2Fresults.md)


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
To know the latest changes in the library please refer to the [changes document](docs/CHANGELOG.md).


## Authors
---

This package has been developed by:

<a href="https://github.com/J0s3M4rqu3z" target="blank">Jose Maria Marquez Blanco</a>
<br/>
<a href="https://www.linkedin.com/in/joancerretani/" target="blank">Joan Alberto Cerretani</a>
<br/>
<a href="https://www.linkedin.com/in/ingvictordurand/" target="blank">Victor Durand</a>
