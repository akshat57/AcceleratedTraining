from reading_datasets import read_ud_dataset
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from textblob import TextBlob
import numpy as np

def read_tb_gum():
    tb_location = '../Datasets/POSTagging/Tweebank/'
    train_tb = read_ud_dataset(dataset = 'tb', location = tb_location, split = 'train')
    dev_tb = read_ud_dataset(dataset = 'tb', location = tb_location, split = 'dev')
    test_tb = read_ud_dataset(dataset = 'tb', location = tb_location, split = 'test')

    gum_location = '../Datasets/POSTagging/GUM/'
    train_gum = read_ud_dataset(dataset = 'gum', location = gum_location, split = 'train')
    dev_gum = read_ud_dataset(dataset = 'gum', location = gum_location, split = 'dev')
    test_gum = read_ud_dataset(dataset = 'gum', location = gum_location, split = 'test')

    return train_tb, dev_tb, test_tb, train_gum, dev_gum, test_gum


def token_number_distribution(train_tb, train_gum):
    #token distribution
    lengths_gum = [len(tokens) for tokens, labels in train_gum]
    lengths_tb = [len(tokens) for tokens, labels in train_tb]

    sns.histplot(lengths_gum, label='GUM', norm_hist = True)
    sns.histplot(lengths_tb, label = 'TB', norm_hist = True)

    plt.legend()
    plt.savefig('logs/token_distribution.jpg')

def calculate_pos_tag_distribution(data):
    pos_counter = {}
    for tokens, labels in data:
        for pos in labels:
            if pos not in pos_counter:
                pos_counter[pos] = [1]
            else:
                pos_counter[pos][0] += 1

    total = sum([pos_counter[pos][0] for pos in pos_counter] )
    for pos in pos_counter:
        pos_counter[pos][0] /= total

    return pos_counter


def pos_tagging_distribution(train_tb, train_gum):
    tb_pos_tags = pd.DataFrame.from_dict(calculate_pos_tag_distribution(train_tb))
    gum_pos_tags = pd.DataFrame.from_dict(calculate_pos_tag_distribution(train_gum))

    sns.set_color_codes('pastel')
    sns.barplot(data = tb_pos_tags, label = 'tb', color = 'k', edgecolor = 'w', orient = 'h')
    sns.set_color_codes('muted')
    sns.barplot(data = gum_pos_tags, label = 'gum', color = 'b', edgecolor = 'w', orient = 'h', alpha = 0.5)
    plt.legend()
    plt.savefig('logs/pos_distribution.jpg')

def calculate_features(train_tb, train_gum):
    
    sentiments = []
    subjectivity = []
    for i, (tokens, labels) in enumerate(train_gum):
        text = ' '.join(tokens)
        blob = TextBlob(text)

        sentiments.append(blob.sentiment_assessments.polarity)
        subjectivity.append(blob.sentiment_assessments.subjectivity)

    sentiments_gum = np.array(sentiments)
    subjectivity_gum = np.array(subjectivity)

    '''print('GUM')
    print('SENTIMENT:', np.mean(sentiments), np.std(sentiments))
    print('SUBJECTIVITY:', np.mean(subjectivity), np.std(subjectivity))'''

    sentiments = []
    subjectivity = []
    for i, (tokens, labels) in enumerate(train_tb):
        text = ' '.join(tokens)
        blob = TextBlob(text)

        sentiments.append(blob.sentiment_assessments.polarity)
        subjectivity.append(blob.sentiment_assessments.subjectivity)

    sentiments_tb = np.array(sentiments)
    subjectivity_tb = np.array(subjectivity)
    
    '''print()
    print('TB')
    print('SENTIMENT:', np.mean(sentiments), np.std(sentiments))
    print('SUBJECTIVITY:', np.mean(subjectivity), np.std(subjectivity))'''

    sns.distplot(subjectivity_gum, label = 'GUM')
    sns.distplot(subjectivity_tb, label = 'TB')
    plt.legend()
    plt.savefig('logs/subjectivity.jpg')



    

if __name__ == '__main__':

    train_tb, dev_tb, test_tb, train_gum, dev_gum, test_gum = read_tb_gum()
    #token_number_distribution(train_tb, train_gum)
    #pos_tagging_distribution(train_tb, train_gum)
    calculate_features(train_tb, train_gum)

    #look at distribution of each POS tag
    #look at distribution of POS tag per sentence
    #look at vocabulary
    #can we convert vocabularies by looking at their POS tags
    #do sentiment analysis
    #do topic modelling
    #think of other ways that can convert you can measure similarity of corpus

    #CREATE A CORPUS SIMILARITY MEASURE USING SOME OF THESE FACTOR
    #DO SCORING BY TAKING A SUBSET OF TRAIN SET, AND MEASURE WITH TEST AND VAL SET
    #SEE IF MAKING GUM DISTRIBUTION CLOSER TO TB HELPS IN PERFORMANCE





    


