# NAE KNN CLassifier
## Functions: 

MODEL TRAINING
train(database) 
INPUT: .csv of data including [filename, label, r, g, b]
OUTPUT: saves knn model (knn_model.sav) and scaler transform (scaler_transform.sav)

CLASSIFY DATA POINT
classify(data)
INPUT: single data point [filename, label, r, g, b]
OUTPUT: prints predicted class based on previously saved model

## Models:
Supports HSV polar, HSV cartesian, RGB 3D, RGB 2D projection

# Requirements

* `python3`
* `pip`
* `virtualenv`

