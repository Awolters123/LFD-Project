# Learning From Data Project Repository 

This repository contains all the code used for our project, it consists of the following files:
- data folder, which contains the the train, dev, and test files, as well as the marked versions. It also contains the used lexicons for creation of the masked dataset, and finally it has the glove_reviews.json file, which is mandatory if you want to run the LSTM model.

- lfd_project.py, this is the python3 file which runs all the models we used for our project with terminal arguments. It can be used with the normal and masked data or with new data in the same format. To run this python script, the following three libraries have to be installed: transformers, sentencepiece, and emoji. To install these run the following commands before executing the script: 
    - pip install transformers
    - pip install sentencepiece
    - pip install emoji
    
    To reproduce our results for each experiment, execute the following commands in the terminal
    (A. trained/tested with standard data, B. trained and tested with masked data, C. trained with masked and tested on standard data):
    
    SVM baseline
    - A. python3 lfd_project.py -i train.tsv -d dev.tsv -t test.tsv -svm_base
    - B. python3 lfd_project.py -i train_mask.csv -d dev_mask.csv -t test_mask.csv -svm_base
    - C. python3 lfd_project.py -i train_mask.csv -d dev_mask.csv -t test.tsv -svm_base
    
    SVM optimised:
    - A. python3 lfd_project.py -i train.tsv -d dev.tsv -t test.tsv -svm_opt
    - B. python3 lfd_project.py -i train_mask.csv -d dev_mask.csv -t test_mask.csv -svm_opt
    - C. python3 lfd_project.py -i train_mask.csv -d dev_mask.csv -t test.tsv -svm_opt
    
    LSTM:
    - A. python3 lfd_project.py -i train.tsv -d dev.tsv -t test.tsv -e glove_reviews.json -lstm -lr 1e-3 -bs 32 -sl 50
    - B. python3 lfd_project.py -i train_mask.csv -d dev_mask.csv -t test_mask.csv -e glove_reviews.json -lstm -lr 1e-3 -bs 32 -sl 50
    - C. python3 lfd_project.py -i train_mask.csv -d dev_mask.csv -t test.tsv -e glove_reviews.json -lstm -lr 1e-3 -bs 32 -sl 50
    
    BERT (bert-base-uncased):
    - A. python3 lfd_project.py -i train.tsv -d dev.tsv -t test.tsv -tf bert -lr 1e-5 -bs 32 -sl 60 -epoch 2
    - B. python3 lfd_project.py -i train_mask.csv -d dev_mask.csv -t test_mask.csv -tf bert -lr 1e-5 -bs 32 -sl 60 -epoch 2
    - C. python3 lfd_project.py -i train_mask.csv -d dev_mask.csv -t test.tsv -tf bert -lr 1e-5 -bs 32 -sl 60 -epoch 2
    
    RoBERTa (roberta-base):
    - A. python3 lfd_project.py -i train.tsv -d dev.tsv -t test.tsv -tf roberta -lr 1e-5 -bs 32 -sl 80 -epoch 2
    - B. python3 lfd_project.py -i train_mask.csv -d dev_mask.csv -t test_mask.csv -tf roberta -lr 1e-5 -bs 16 -sl 80 -epoch 2
    - C. python3 lfd_project.py -i train_mask.csv -d dev_mask.csv -t test.tsv -tf roberta -lr 1e-5 -bs 16 -sl 80 -epoch 2
    
    DeBERTa (microsoft/deberta-v3-base):
    - A. python3 lfd_project.py -i train.tsv -d dev.tsv -t test.tsv -tf deberta -lr 1e-5 -bs 32 -sl 70 -epoch 2
    - B. python3 lfd_project.py -i train_mask.csv -d dev_mask.csv -t test_mask.csv -tf deberta -lr 1e-5 -bs 32 -sl 70 -epoch 2
    - C. python3 lfd_project.py -i train_mask.csv -d dev_mask.csv -t test.tsv -tf deberta -lr 1e-5 -bs 32 -sl 70 -epoch 2

- results.pdf, this file contains the results we obtained by running the models with the described dataset and script. It has the classifcation report and confusion matrix displayed for each models experiment on the test set (1. SVM baseline, 2. SVM optimised, 3. LSTM, 4. BERT, 5. RoBERTa, and 6. DeBERTa).



