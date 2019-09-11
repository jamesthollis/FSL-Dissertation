import json
import os
import tarfile
import numpy as np
import re
import unicodedata
import csv
import editdistance
import nltk
from nltk.corpus import stopwords

def load_batch(batch_filepath, user_info, lexicon):
    print('Batch {}'.format(batch_filepath[-2:]))
    filenames = []
    for r, d, f in os.walk(batch_filepath):
        for filename in f:
            if '.tweets' in filename:
                filenames.append(os.path.join(r, filename))
    
    cd_ids, cp_ids, dp_ids = [], [], []
    cd_labels1, cp_labels1, dp_labels1 = [], [], []
    x, last_user = 0, ' '
    with open(user_info, 'r') as label_info:
        for line in label_info:
            csv_row = line.split(',')
            if x == 1 and last_user == 'depression':
                cd_ids.append(csv_row[0])
                cd_labels1.append('control')
                x = 0
            elif x == 1 and last_user == 'ptsd':
                cp_ids.append(csv_row[0])
                cp_labels1.append('control')
                x = 0
            elif csv_row[4] == 'depression':
                cd_ids.append(csv_row[0])
                cd_labels1.append('depression')
                dp_ids.append(csv_row[0])
                dp_labels1.append('depression')
                x = 1
                last_user = 'depression'
            elif csv_row[4] == 'ptsd':
                cp_ids.append(csv_row[0])
                cp_labels1.append('ptsd')
                dp_ids.append(csv_row[0])
                dp_labels1.append('ptsd')
                x = 1
                last_user = 'ptsd'

    cd_users, cp_users, dp_users = [], [], []
    cd_labels, cp_labels, dp_labels = [], [], []
    for file in filenames:
        user_tweets = []
        user_id = file[34:-7]
        with open(file,'r') as tweet_collection:
            for line in tweet_collection:
                tweet = json.loads(line)
                user_tweets += process_tweet(tweet)['tokens']
        user_tweets = list(filter(None, user_tweets))
        
        if user_id in cd_ids:
            cd_users.append(user_tweets)
            if cd_labels1[cd_ids.index(user_id)] == 'control':
                cd_labels.append(0)
            else:
                cd_labels.append(1)
        if user_id in cp_ids:
            cp_users.append(user_tweets)
            if cp_labels1[cp_ids.index(user_id)] == 'control':
                cp_labels.append(0)
            else:
                cp_labels.append(1)
        if user_id in dp_ids:
            dp_users.append(user_tweets)
            if dp_labels1[dp_ids.index(user_id)] == 'depression':
                dp_labels.append(0)
            else:
                dp_labels.append(1)
                
    return cd_users, cp_users, dp_users, cd_labels, cp_labels, dp_labels

def unzip_batch(zip_filepath, target_directory, zip_type="r:gz"):
    
    tar = tarfile.open(zip_filepath, zip_type)
    tar.extractall(path=target_directory)
    tar.close()

def word_tokenize(tweet):
    return tweet.split(" ")

def remove_emojis(tweet):
    emojis = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    no_emoji_tweet = []
    for word in tweet:
        word = emojis.sub(r'', word)
        no_emoji_tweet.append(word)
    return no_emoji_tweet

def remove_symbols(tweet):
    no_symbol_tweet = []
    for word in tweet:
        word = re.sub(r':', '', word)
        word = re.sub(r'‚Ä¶', '', word)
        word = re.sub(r'[^\x00-\x7F]+',' ', word)
        word = re.sub(r'@[a-zA-Z0-9]*','name', word)
        word = re.sub(r'\.\?\!\'\"', '', word)
        word = re.sub(r'^https?:\/\/.*[\r\n]*', '', word)
        no_symbol_tweet.append(word)
    return no_symbol_tweet
    
def repeated_letters(tweet):
    no_repeats_tweet = []
    for word in tweet:
        word = re.sub(r'(.)\1{2,}', r'!', word)
        no_repeats_tweet.append(word)
    return no_repeats_tweet

def tokenize(tweet):
    token_data = word_tokenize(tweet)
    token_data = remove_emojis(token_data)
    token_data = remove_symbols(token_data)
    token_data = repeated_letters(token_data)
    filtered_tweet = [w for w in token_data]
    return filtered_tweet
    
def remove_stopwords(tweet):
    stop_words = set(stopwords.words('english'))
    no_stop_tweet = []
    for word in tweet:
        if word not in stop_words:
            no_stop_tweet.append(word)
    return no_stop_tweet  
    
def remove_emoticons(tweet):
    emoticons = set([
        ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
        ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
        '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
        'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
        '<3', ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
        ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
        ':c', ':{', '>:\\', ';('
        ])
    no_emoticon_tweet = []
    for word in tweet:
        if word not in emoticons:
            no_emoticon_tweet.append(word)
    return no_emoticon_tweet
    
def spelling_score(tweet, lexicon):
    corrected_tweet = []
    spelling_score = 0
    for word in tweet:
        best_distance = 10000
        best_word = ''
        for reference in lexicon:
            current_score = editdistance.eval(word, reference)
            if current_score == 0:
                best_word = reference
                best_distance = 0
                break
            elif current_score < best_distance:
                best_word = reference
                best_distance = current_score
        if best_distance < 6:
            spelling_score += best_distance
            corrected_tweet.append(best_word)
        
    normalised_score = spelling_score/len(tweet)
    return corrected_tweet, normalised_score
    

def preprocess_tweet(all_tweet, lexicon):
    if 'text' in all_tweet and all_tweet['text']:
        tweet = all_tweet['text'][2:-1]
    else:
        print(all_tweet)
        tweet = ''
    string_tweet = str(unicodedata.normalize('NFKD', tweet).encode('ascii','ignore'))
    lowercase_tweet = string_tweet.lower()
    token_tweet = tokenize(lowercase_tweet)
    no_stop_tweet = remove_stopwords(token_tweet)
    no_emoticons_tweet = remove_emoticons(no_stop_tweet)
    corrected_tweet, tweet_score = spelling_score(no_emoticons_tweet, lexicon)
    string_tweet = ' '.join(corrected_tweet)
    lowercase_tweet = string_tweet.lower()
    all_tweet['text'] = lowercase_tweet
    all_tweet['score'] = tweet_score
    
    return all_tweet
