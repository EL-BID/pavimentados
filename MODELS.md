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
