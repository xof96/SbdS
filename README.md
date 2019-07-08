# Search By Description Similarity (SBDS)

Search of images given a description models trained over COCO dataset subset

## Preparing the data

The data must be downloaded from <br> 
https://drive.google.com/drive/folders/1RzGYR2uqMRS4WqX_wqIiI2Y_NdNAey1m 
<br> and unzipped inside a data/ folder which has
to be right inside the root folder. There are 3 test folders, so the final 
structure must be like the following:

````
SBDS
    |-data
        |-test_A_data
        |-test_A_images
        |-test_B_data
        |-test_B_images
        |-test_C_data
        |-test_C_images
        |-train_data
        |-train_images
````


## Preparing environment
A env.py file must be created at project root folder level and must contain 
environmental variables, the format is the following:

```python
import os
ABS_PATH = '<path_to_project>/SbdS'
DATA_PATH = 'data'

TRAIN_DATA = os.path.join(ABS_PATH, DATA_PATH, 'train_data')
TRAIN_IMAGES = os.path.join(ABS_PATH, DATA_PATH, 'train_images')
TEST_A_DATA = os.path.join(ABS_PATH, DATA_PATH, 'test_A_data')
TEST_A_IMAGES = os.path.join(ABS_PATH, DATA_PATH, 'test_A_images')
TEST_B_DATA = os.path.join(ABS_PATH, DATA_PATH, 'test_B_data')
TEST_B_IMAGES = os.path.join(ABS_PATH, DATA_PATH, 'test_B_images')
TEST_C_DATA = os.path.join(ABS_PATH, DATA_PATH, 'test_C_data')
TEST_C_IMAGES = os.path.join(ABS_PATH, DATA_PATH, 'test_C_images')
```
So it is very important to replace the **ABS_PATH** information.