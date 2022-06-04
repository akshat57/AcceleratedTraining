import pickle
import json

def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

def read_pos_standard(filename, sep = '\t'):
    '''
    Reads the standard POS tagging input data format with token SEP label format. Utterance are separated by '\n'. SEP token is input to the function.
    INPUTS:
        filename: reading file name
        sep: separator token between token and labels in the standard format. Usually '\t' or space.  
    '''

    data = []
    tokens = []
    labels = []
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            line = line[:-1]

            if len(line):
                token, label = line.split(sep)
                tokens.append(token)
                labels.append(label)

            else:
                data.append((tokens, labels))
                tokens = []
                labels = []

    return data



def read_gimpel():
    location = 'Gimpel/'
    filename = location + 'oct27.conll'

    data = read_pos_standard(filename)

    return data


def read_ritter():
    location = 'Ritter/'
    filename = location + 'train'

    data = read_pos_standard(filename)

    return data

def read_ud_dataset(dataset = 'tb', location = '../Datasets/POSTagging/Tweebank/', split = 'train'):
    if dataset == 'tb':
        filename = location + 'en-ud-tweet-' + split + '.conllu'
    elif dataset == 'gum':
        filename = location + 'en_gum-ud-' + split + '.conllu'

    data = []
    tokens = []
    labels = []
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            line = line[:-1]

            if len(line) and line[0] != '#':
                line = line.split('\t')
                index = line[0]
                if index.find('-') == -1:
                    tokens.append(line[1])
                    labels.append(line[3])

            elif len(line) == 0:
                data.append((tokens, labels))
                tokens = []
                labels = []

    return data

def reverse_dictionary(input_dict):
    output_dict = {}
    
    for key in input_dict:
        output_dict[input_dict[key] ] = key
        
    return output_dict

def read_conll2003():
    tag_mapping = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
 'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
 'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
 'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
 'WP': 44, 'WP$': 45, 'WRB': 46}
    
    index_to_pos = reverse_dictionary(tag_mapping)
    
    
    location = 'CONLL2003/'
    filename = location + 'conll2003_train.pkl'

    saved_data = load_data(filename)
    data = []
    for i, line in enumerate(saved_data):
        labels = [index_to_pos[label] for label in line['pos_tags']]
        data.append( (line['tokens'], labels) )
        
    return data


def get_all_labels(data):
    all_labels = []
    for tokens, labels in data:
        all_labels += labels

    all_labels = set(all_labels)
    return all_labels

def read_lexnorm(location = '../Datasets/WNUTlexnorm2015/'):
    filename = location + 'train_data.json'
    f = open(filename)
    data = json.load(f)

    inorm_dict = {}

    for row in data:
        for input_token, output_token in zip(row['input'], row['output']):
            input_token = input_token.lower()
            output_token = output_token.lower()

            if input_token != output_token:
                if output_token not in inorm_dict:
                    inorm_dict[output_token] = [input_token]
                else:
                    inorm_dict[output_token].append(input_token)

    for key in inorm_dict:
        inorm_dict[key] = list(set(inorm_dict[key]))

    return inorm_dict

    

if __name__ == '__main__':
    #data = read_ud_dataset(dataset = 'gum', location = '../Datasets/POSTagging/GUM/', split = 'dev')
    #print(len(data))

    read_lexnorm()




