# Goal

The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

## Guidlines

- Everything must be represented in python notebooks for a human to be understandable, add comments, graphics etc.
- The notebok should do everything step by step, if something doesn't work well optimize in the next cell with comments-
- Add markdown blocks for description what you did
- Use always latest library version, update via uv if necessary
- Use fastai library if necesary, but feel free to tune underlying parameters, do not just use defaults
- Run training rounds with small models step by step, run them locally
- For larger models and GPU heavy tasks push to kaggle
- Important decisions must be logged to decision_log.md

## STEP1 DATA Cleaning

- Inspect the data in train.csv (do NOT Read test.csv) via a python notebook (data_cleaning.ipynb)
- TEST.csv does not contain surviver data but is for submission only.
- Data is described in data_description.md
- Go step by step, do regular Data analytics tasks
- find outliers
- Find dependencies betweeen the columns
- Generate feature columns for future processing
- Think about an optimized representation of the data
- Choose a good representative split (train and evaluation) for the data (use 42 as seed, if random is necessary).
- Save the cleaned data as new csvs (train and eval)


## STEP2 Model training

- Train a model using the cleaned data, test the data on test.csv. Goal is to optimize the prediction of Survial rate (percentage of correct predicted answers)
- NEVER directly Read the test.csv file, only use it for evaluation
- Optimize hyper parameters and traing data optimization with known best practices
- Generate