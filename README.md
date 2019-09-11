# FSL-Dissertation

Code for dissertation on Few-Shot Learning for Predicting Mood using Online Data

The code for this project consists of 2 files, load_tweets.py and model.ipynb.

For the code to run successfully, the CLPsych data should be saved in a folder "./training_data" which contains zipped folders representing each batch. The code unzips these folders, which contain a document for each user.

Additionally, this code needs the twitter lexicon in the same directory as the code, saved as "./twitter-lexicon.txt". The other dependency is the twitter embeddings, which should be saved in a folder in the directory titled "./embeddings".

The model file utilises the preprocessing in load_tweets.py, and outputs the predictions and classification reports for each model. These can then be used to generate the results and graphs which are presented in the dissertation report.
