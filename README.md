# CNN-T-DB
## Description

This work aims to build a database asisted tissue texture from full dose computed tomography (CT) for low dose CT image reconstruction. There are main two contributions.
- One contribution is to learn tissue-specific texture using the convounitional neural network (CNN-T) for each patient on slice level.
- The other contribution is to find an appropriate CNN-T from database for any coming LdCT subject using multi-modality feature selection model. This model is built using the randomforest in R package.

## CNN-T model 
- Input 
  - The extracted patches with size 7by7 from the lung region is stored in the folder patches.
- Output
  - The trained model is stored in the folder models
- Sourece code 
  - build_model.py -> to build the network structure
  - input_data_triple.py -> to load the data
  - train.py -> to train the model 

## Candidate selection model using multi-modality features
- Input: Features from three modalities are used.
  - subjects' physicological factors, e.g. body mass index (BMI), age, gender
  - the CT scan protocal, e.g. position, features extracted from CT images 
  - a novel feature named Lung Mark, which is deliberatelty proposed to reflect the z-axial property of human anatomy
- Sourece code
  - MRFDataBasePF.R -> To train and test the selection model 
