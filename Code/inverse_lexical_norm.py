from useful_functions import load_data, save_data
from sklearn import metrics
from analyse_data import read_tb_gum
import random
import numpy as np
from symbols import twitter_symbols
from reading_datasets import read_lexnorm

def inorm_propn(data, multiplier = 1):
    #Do inverse lexical normalization for PROPN with @user1234 and hashtags
    
    augmented_data = []
    for tokens, labels in data:
        new_tokens, new_labels = [], []
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            new_token = token

            if label == 'PROPN':
                #Replacing with @user1234
                if random.random() < 0.25 * multiplier:
                    username = '@USER' + str(random.randint(100, 9999))
                    new_token = username
                #If not replacing with user, convert to hashtag
                elif random.random() < 0.1 * multiplier:
                    new_token = '#' + token

            new_tokens.append(new_token)
            new_labels.append(label)

        augmented_data.append((new_tokens, new_labels) )

    return augmented_data


def collect_hashtags():
    #Collect all hashtags from the unlabelled TB train set

    train_tb, dev_tb, test_tb, train_gum, dev_gum, test_gum = read_tb_gum()

    hashtags = []
    for tokens, labels in train_tb:
        for token, label in zip(tokens, labels):
            if token[0] == '#':
                hashtags.append(token)

    return hashtags



def inorm_X(data, multiplier = 1):
    '''
        Do inverse lexical normalization for X with RT, @user1234, URL and hashtags

        REPLACEMENT NOTES:
        RT : 0.2490
        @USER1234 : 0.2539
        URL1234 : 0.2983
        #hashtags : 0.1481

        This accounts for about 95% of the X symbols
    '''

    hashtags = collect_hashtags()

    augmented_data = []
    for tokens, labels in data:

        #Add RT, username to the beginning of the sentence
        if random.random() < 0.3 * multiplier:
            username = '@USER' + str(random.randint(100, 9999))

            tokens.insert(0, username)
            tokens.insert(0, 'RT')
            labels.insert(0, 'X')
            labels.insert(0, 'X')

        #Add URL to the end of the sentence
        if random.random() < 0.3 * multiplier:
            url = 'URL' + str(random.randint(10, 9999))
            tokens.append(url)
            labels.append('X')

        #Add random hashtags from the unlabelled train set
        if random.random() < 0.2 * multiplier:
            index = int( min( max(0, np.random.normal(0.8070, 0.266, 1) ), 1) * len(tokens) )
            selected_hashtag = random.choice(hashtags)

            tokens.insert(index, selected_hashtag)
            labels.insert(index, 'X')

        augmented_data.append((tokens, labels))

    return augmented_data



def inorm_sym(data, twitter_symbols, threshold = 0.25):
    #Add twitter like symbols in randomly

    twitter_symbols = list(set(twitter_symbols))
    augmented_data = []

    for tokens, labels in data:
        #Add random twitter symbols from the unlabelled train set
        if random.random() < threshold:
            index = int( min( max(0, np.random.normal(0.771, 0.269, 1) ), 1) * len(tokens) )
            selected_symbol = random.choice(twitter_symbols)

            tokens.insert(index, selected_symbol)
            labels.insert(index, 'SYM')

        augmented_data.append((tokens, labels))

    return augmented_data

def do_ilexnorm(data, threshold = 0.8):
    #Do inverse lexcial normalization
    inorm_dict = read_lexnorm()
    augmented_data = []

    word_count = 0
    sent_count = 0
    for tokens, labels in data:
        new_tokens, new_labels = [], []
        
        sent_flag = True
        for i, (token, label) in enumerate(zip(tokens, labels)):
            new_token = token.lower()

            if new_token in inorm_dict:
                #do lexical un-normalization with [THRESHOLD]% of the times the word is found in dictionary
                if random.random() < threshold:
                    new_token = random.choice(inorm_dict[new_token])
                    word_count += 1

                    if sent_flag:
                        sent_count += 1
                        sent_flag = False

            
            new_tokens.append(new_token)
            new_labels.append(label)

        augmented_data.append((new_tokens, new_labels) )

    return augmented_data


def inverse_lexical_norm(data, twitter_symbols):
    #propn_data = inorm_propn(data)
    #X_data = inorm_X(propn_data)
    #sym_data = inorm_sym(X_data, twitter_symbols)
    
    norm_data = do_ilexnorm(data)

    return norm_data


if __name__ == '__main__':

    train_tb, dev_tb, test_tb, train_gum, dev_gum, test_gum = read_tb_gum()

    #augmented_data = inverse_lexical_norm(train_gum, twitter_symbols)

    for multiplier in [1, 2, 3]:
        augmented_data = inorm_X(train_gum, multiplier = 1)
        save_data('../Datasets/POSTagging/GUM_augemented/train_GUM_X_' + str( int(multiplier) ) + '.pkl', augmented_data)
