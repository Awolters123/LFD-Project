# Learning From Data Project Repository 

This repository contains all the code used for our project, it consists the following files:
- data folder, which contains the the train, dev, and test files, as well as the marked versions. It also contains the used lexicons for creation of the masked dataset, and finally it has the glove_reviews.json file, which is mandatory to run the LSTM model.

- lfd_project.py, this is the python3 file which runs all the models we used for our project with terminal arguments. It can be used with the normal and masked data. To run this python script, the following three libraries have to be installed: transformers, sentencepiece, and emoji. To install these run the following command before executing the script: 
    pip install transformers
    pip install sentencepiece
    pip install emoji

- results.pdf, this file contains the results we obtained by running the models with the described dataset and script. It has the classifcation report and the confusion matrix for each model (1. SVM baseline, 2. SVM optimised, 3. LSTM, 4. transformers: bert, roberta, and deberta).



