# Search By Description Similarity (SBDS)

Search of images given a description models trained over COCO dataset subset

## Preparing the data

The data must be downloaded from <br> 
https://drive.google.com/drive/folders/1RzGYR2uqMRS4WqX_wqIiI2Y_NdNAey1m 
<br> and unzipped inside a data/ folder which has
to be right inside the root folder. The test folders and files must be renamed 
from _test_A_*_ to _test_*_

## Preparing environment
A env.py file must be created at project root folder level and must contain 
environmental variables, the format is the following:

```python
import os
ABS_PATH = '<path_to_project>/SbdS'
DATA_PATH = 'data'

TRAIN_DATA = os.path.join(ABS_PATH, DATA_PATH, 'train_data')
TRAIN_IMAGES = os.path.join(ABS_PATH, DATA_PATH, 'train_images')
TEST_DATA = os.path.join(ABS_PATH, DATA_PATH, 'test_data')
TEST_IMAGES = os.path.join(ABS_PATH, DATA_PATH, 'test_images')
```
So it is very important to replace the **ABS_PATH** information.