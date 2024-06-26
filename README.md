This repository is for COMP6258 coursework regarding the paper"Rotating Features for Object Discovery" ([https://openreview.net/pdf?id=fg7iyNK81W](https://openreview.net/pdf?id=fg7iyNK81W)).


First,  please follow the instruction of "RotatingFeatures-main/README.m" to build the basic environment.

To reproduce these experiments, then you need to 

# Different Dataset

* Please add files under folder "ADE20K2017" and "MNISTShape" to the path "RotatingFeatures-main/datasets". 
* Please add files "ADE20KDataset.py" and "MNISTShape.py" to the path "RotatingFeatures-main/codebase/data". 
* Please add config files "ADE20K.yaml" , "MNISTShape.yaml" to the path "RotatingFeatures-main/codebase/config/experiment".
* replace "data_utils.py" to the same file under the folder "RotatingFeatures-main/codebase/utils"
* To use ADE20K, use "python -m codebase.main +experiment=ADE20K". 
* To use MNISTShape, use "python -m codebase.main +experiment=MNISTShape".

# Different cluster

* Please replace the files"eval_utils.py" and "rotation_utils.py" under folder"Different cluster" to the same files under this folder "RotatingFeatures-main/codebase/utils". 
* Replace the file "4Shapes.yaml" to the same file under the  folder "RotatingFeatures-main/codebase/". 
* Run the code,  use "python -m codebase.main +experiment=4Shapes" 

# Different Optimisers

* To run “Different Optimisers” experiment, copy the "model_utils.py" file in the "Different optimiser" folder into "RotatingFeatures-main/codebase/utils" path and replace it with the existing file.
* Then uncomment lines 11 to 69 to run Adam optimiser, lines 72 to 143 to run SGD optimiser, and lines 145 to 182 to run Adadelta optimiser.

# Test Robustness

* To run "Testing Robusteness" experiment, copy the "ShapesDataset.py" file in the "Testing Robusteness" folder into "RotatingFeatures-main/codebase/data" path and replace it with the existing file.
* Then uncomment lines 54 to 58 to run a for Loop for label flipping attack.

# Different Model Architecture

* To run experiments with different model architectures, replace the file	RotatingFeatures-main/codebase/model/RotatingAutoEncoder.py with Different_architecture/RotatingAutoEncoder.py 


* To experiment with adding dropout in encode replace the RotatingFeatures-main/codebase/model/ConvEncoder.py with Different_architecture/ConvEncoder.py 
