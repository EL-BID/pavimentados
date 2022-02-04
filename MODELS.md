# Pavimentados system models
---

As we saw above, the system has mainly 4 models. The pavement failure detection model, the signage detection model, the signage classification model and the signage status identification model. We will see in detail each of the models below.

## Detection of pavement failures
---

This model is based on Yolo version 3, which allows to identify objects or elements within an image. It was retrained to detect 8 types of faults:

1. Longitudinal Linear Cracks: these are cracks in the pavement that are longitudinally located.

2. Longitudinal Linear Intervals: These are cracks in the pavement that represent joints of sections and are not faults.

3. Transverse Linear Cracks: These are cracks in the pavement that are transversely located.

4. Transverse Linear Intervals: These are cracks in the pavement that represent joints of sections and are not faults.

5. Crocodile skin: These are clusters of fissures that are usually found in blocks in such a way that they look like the scaly skin of a crocodile.

6. Protrusions or Potholes: These are holes found in the pavement that may or may not be scaled.

7. White line blur: Occurs when the delineation or demarcation on the pavement is out of focus or blurred.

8. Other faults: It is the grouping of all the other faults that could not be grouped in the previous ones.

The model returns the following information:

 - Class of the identified object
 - Location within the image
 - Probability with which that detection is considered

## Signal detection
---

This model is based on Yolo version 3, which allows to identify objects or elements within an image. It was retrained to detect 1 type of element, and is capable of finding signage on the path.

The output of the model is similar to what you get from the pavement fault detection model only that it has a single output class.

## Classification of signage
---

This model has a Siamese architecture, where we focus on comparing two images, in this case the images of known signals from our database with the signal found in the image.

The main classes are the followingÑ

1. Warning signs (yellow diamonds)
2. Regulatory signs (red circles with or without prohibitions).
3. Stop signs
4. Information signs (blue signs with route information).
5. Information signs (brown signs with information about the route).
6. Road information signs (green signs with specifics of the route to be followed).
7. White signs with specific information of the area.
8. Other

## Condition of the signage
---

This model identifies whether the signal is in good or bad condition, or if the signal is rotated so that no status can be determined on that image.