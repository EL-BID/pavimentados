![analytics image (flat)](https://raw.githubusercontent.com/vitr/google-analytics-beacon/master/static/badge-flat.gif)
![analytics](https://www.google-analytics.com/collect?v=1&cid=555&t=pageview&ec=repo&ea=open&dp=/pavimentados/readme&dt=&tid=UA-4677001-16)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=EL-BID_pavimentados&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=EL-BID_pavimentados)
[![Downloads](https://pepy.tech/badge/pavimentados)](https://pepy.tech/project/pavimentados)
# Pavimentados

## Project Description
---

Pavimentados is a tool that allows the identification of pavement faults located on highways or roads, as well as vertical and horizontal signage. This library is an environment around the computer vision models developed to allow the detection of the different elements. The faults and detected elements are then used to generate metrics that help in road maintenance planning.

To use it you must contact the Inter-American Development Bank to obtain the credentials that give access to the models with which the API works.

These models require images or videos taken with the specifications that will be explained in later sections. 

So far the system uses 4 models involved in different phases of detection and categorization.

| Model Name             | Description                                         | Classes |
|------------------------|---------------------------------------------------- | ------- |
| Paviment failures      | Detection of failures on the road and classifies it | 8       |
| Signage detection      | Detection of signage on the road                    | 2       |
| Signage classification | Classifies the signage detected                     | 314     |
| Signage State          | Detect if the signage is in good or bad state       | 3       |

To understand each model in detail see the [models](https://github.com/EL-BID/pavimentados/blob/main/MODELS.md) section.

## Main Features
---

Some of the features now available are as follows:

- Scoring using the models already developed
- Workflows for image acquisition and assessment
- Evaluation of gps information to determine the location of faults and elements.
- Image or video support
- Support for GPS data in different formats (GPRRA, csv, embedded in image)
- Download models directly into the root of the package

## Instalation
---

To install you can use the following commands

```
pip install pavimentados
```

Then to download the models use the following commands

```
from pavimentados import download_models

download_models(url = <signed_url>)
```

Or alternativily

```
from pavimentados import download_models

download_models(aws_access_key = <aws_access_key>, signature = <signature>, expires = <expiration_time>)
```


To obtain the corresponding credentials for downloading the models, please contact the Inter-American Development Bank at infradigital@iadb.org

You can also clone the repository but remember that the package is configured to download the models and place them in the root of the environment.

## Quick Start
---

To make use of the tool import the components that create a workflow with images

```
from pavimentados.processing.processors import MultiImage_Processor
from pavimentados.processing.workflows import Workflow_Processor
```

In this example, we have the image processor object MultiImage_Processor, which is in charge of taking the images and analyzing them individually using the models. In addition, we have the Workflow_Processor object that is in charge of the image processing workflow. 

Internally, the Workflow_Processor has objects that can interpret different image sources or GPS information. 

Among the allowed image sources we have:

 - image_routes: A list of image routes
 - image_folder: a folder with all images
 - images: images already loaded in numpy format
 - video: The path to a video file

Among the allowed GPS data sources we have:

 - image_routes: A list of the routes of the images that have the gps data embedded in them.
 - image_folder: A folder with all the images that have the gps data embedded in them.
 - loc: A file in GPRRA format
 - csv: A gps file with the gps information in columns and rows.

Once these elements are imported, the processor is instantiated as follows

```
ml_processor = MultiImage_Processor(assign_devices = True, gpu_enabled = True)
```

The processor has the ability to allocate GPU usage automatically assuming that 6GB is available, it can be parameterized so that it is not automatically allocated, pass the allocation as a parameter, or even not work with the GPU.

You can modify the devices used according to the TensorFlow documentation regarding GPU usage (see https://www.tensorflow.org/guide/gpu)

The workflow object is able to receive the instantiated processor, without it it is not able to execute the workflow.

```
workflow = Workflow_Processor(route, image_source_type='image_folder', gps_source_type = 'image_folder')
```

The complete execution code would be as follows:

```
from pavimentados.processing.processors import MultiImage_Processor
from pavimentados.processing.workflows import Workflow_Processor
from pathlib import Path

### Image with the GPS data embebed
route = Path(<route with the images for processing>)

ml_processor = MultiImage_Processor(assign_devices = True, gpu_enabled = True)

workflow = Workflow_Processor(route, image_source_type='image_folder', gps_source_type = 'image_folder')

results = workflow.execute(ml_processor)
```

In results you will find the following:

 1. table_summary_sections: DataFrame with summary table by sections.
 2. data_resulting: DataFrame with results per frame
 3. data_resulting_fails: DataFrame with results by unique faults encountered.
 4. signals_summary: DataFrame with all the information about the signals.
 5. raw_results: Raw results totals

 ## Autores
---

This package has been developed by:

<a href="https://github.com/J0s3M4rqu3z" target="blank">Jose Maria Marquez Blanco</a>
<br/>
<a href="https://www.linkedin.com/in/joancerretani/" target="blank">Joan Alberto Cerretani</a>
