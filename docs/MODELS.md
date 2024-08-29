# Pavimentados system models
---

As we saw above, the system has mainly 3 models. 
* Pavement failure detection model
* Signage detection model
* Signage classification model 

We will see in detail each of the models below.

## Detection of pavement failures
---

This model is based on Yolo version 8, which allows to identify objects or elements within an image. It was retrained to detect 8 types of faults:

1. Longitudinal Linear Cracks (D00): these are cracks in the pavement that are longitudinally located.
2. Longitudinal Linear Intervals (D01): These are cracks in the pavement that represent joints of sections and are not faults.
3. Transverse Linear Cracks (D10): These are cracks in the pavement that are transversely located.
4. Transverse Linear Intervals (D11): These are cracks in the pavement that represent joints of sections and are not faults.
5. Crocodile skin: These are clusters (D20) of fissures that are usually found in blocks in such a way that they look like the scaly skin of a crocodile.
6. Protrusions or Potholes (D40): These are holes found in the pavement that may or may not be scaled.
7. Cross walk blur (D44): Occurs when the delineation or demarcation on the pavement is out of focus or blurred.
8. White line blur (D43): Occurs when the delineation or demarcation on the pavement is out of focus or blurred.
9. Other faults (OT0): It is the grouping of all the other faults that could not be grouped in the previous ones.

The model returns the following information:

 - Class of the identified object
 - Location within the image
 - Probability with which that detection is considered


## Signal detection
---
This model is based on Yolo version 8, which allows to identify objects or elements within an image. It was retrained to detect 1 type of element, and is capable of finding signage on the path.
The output of the model is similar to what you get from the pavement fault detection model only that it has a single output class.


## Classification of signage
---
This model has a Siamese architecture to find the corresponding signals in the image.
