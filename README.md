# Development and Validation of an anomaly-based detection model for unintentional KCl prescribing errors using machine learning 

The codes for `Model training and testing` in Materials and Methods of the paper are available.

*** Note that all codes are executable ONLY if your own data exist

## Datasets

- Here, the datasets we used in the paper can not be released for personal information protection.

- Instead, you can identify a sample dataset. Please refer to `sample_dataset.csv`
  - `sample_dataset.csv` shows the examples of the datasets used for modeling (Note that this is not a real patients' dataset)
  - `data_type_info.csv` include the variable types (See Appendix 1 if you would like to know the details)

## Modeling

- This code covers the gridsearch process and training with the best params for ANN

- ANN gridsearch was conducted with TensorFlow 2.0

- See `ANN_code.py`

- You can see the tensorboard results of gridsearch with the accuracies and aurocs [here](https://tensorboard.dev/experiment/NirtopfjTyaYPCle9QqHgA/).  
  (this is the same result with `tensorboard --logdir=./logs/hparam_tuning_results` if you run the code with your own datasets)


