# Distribution Thresholder

An classifier for anomaly scores of Time-Series data. Uses hypothesis testing to find the best fitting distribution to the normal anomaly score, and selects a threshold based on a given risk parameter.

## Installation

To install the module, simply run the following command :
```{python3}
pip install .
```



## Usage

The <b>Thresholder</b> class contains two main functions : ``` fit ``` and ``` predict ```
- fit : 

- predict : 

It also contains useful parameters :

- model_ : 

- fit_labels_ : 

- fit_survival_ : 

Example :

``` {python3}
from Distributrion_Thresholder.threshold import Threshold


thesh = Threshold('Anderson')
thresh.fit(training_score)
```
