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

To understand each model in detail see the [models](https://github.com/J0s3M4rqu3z/pavimentados/blob/main/MODELS.md) section.

## Main Features
---

Some of the features now available are as follows:

- Scoring using the models already developed
- Workflows for image acquisition and assessment
- Evaluation of gps information to determine the location of faults and elements.
- Image or video support
- Support for GPS data in different formats (GMMRA, csv, embedded in image)
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
from viasegura import download_models

download_models(aws_access_key = <aws_access_key>, signature = <signature>, expires = <expiration_time>)
```


To obtain the corresponding credentials for downloading the models, please contact the Inter-American Development Bank.

You can also clone the repository but remember that the package is configured to download the models and place them in the root of the environment.