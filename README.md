# Distribution Thresholder

An classifier for anomaly scores of Time-Series data. Uses hypothesis testing to find the best fitting distribution to the normal anomaly score, and selects a threshold based on a given risk parameter.

## Installation

To install the module, simply run the following command :
```{python3}
pip install .
```



## Usage

The <b>Thresholder</b> class contains two main functions : ``` fit ``` and ``` predict ```

- fit : fits the best statistical distribution to the input scores
    - input : 
        - data: 1D-array of anomaly scores

- predict : predicts the labels and the survival density of the input scores, based on the fitted distribution
    - input : 
        -  score: 1D-array of anomaly scores
        - percentage: risk parameter, between 0 and 1 
    - output : dictionary containing the following elements :
        - label: predicted labels for the input score
        - sf: predicted survival density of the input score

It also contains useful attributes :

- model_ : The statistical model chosen by the thresholder

- fit_labels_ : The predicted labels of the scores used for fitting

- fit_survival_ : The predicted survival density of the scores used for fitting

Example :

``` {python3}
import numpy as np
from Distributrion_Thresholder.threshold import Threshold

training_score = np.random.lognormal(mean=0, sigma=1, size=2000)
testing_score = np.random.lognormal(mean=0, sigma=1, size=100)

thesh = Threshold('Anderson')
thresh.fit(training_score)
labels = thresh.predict(testing_score,parameter = 0.05)['label']
```
