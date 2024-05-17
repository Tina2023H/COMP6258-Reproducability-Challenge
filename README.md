This repository is for COMP6258 coursework regarding the paper"Rotating Features for Object Discovery" (https://openreview.net/pdf?id=fg7iyNK81W).


First,  please follow the instruction of "RotatingFeatures-main/README.m" to build the basic environment.

To reproduce these experiments, you need to 

Different Dataset
 1 please add folder "ADE20K2017" and "MNISTShape" to the path "RotatingFeatures-main/datasets". 
 2 please add file "ADE20KDataset.py" and "MNISTShape.py" to the path "RotatingFeatures-main/codebase/data"
 3 please add config files "ADE20K.yaml" , "MNISTShape.yaml" to the path "RotatingFeatures-main/codebase/config/experiment"
 4 to use ADE20K, use "python -m codebase.main +experiment=ADE20K"
   to use MNISTShape, use "python -m codebase.main +experiment=MNISTShape"

Different cluster
 1 please replace the files"eval_utils.py" and "rotation_utils.py" under folder"Different cluster" to the same filed under this folder "RotatingFeatures-main/codebase/utils".
 2 replace the file "4Shapes.yaml" to the same file under the  folder "RotatingFeatures-main/codebase/".
 3 run the code,  use "python -m codebase.main +experiment=4Shapes"
