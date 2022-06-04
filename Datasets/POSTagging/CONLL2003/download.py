from datasets import load_dataset
import pickle

def save_data(filename, data):
    #Storing data with labels
    a_file = open(filename, "wb")
    pickle.dump(data, a_file)
    a_file.close()
    

def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

dataset = load_dataset("conll2003")


train_dataset = []
for data in dataset['train']:
    train_dataset.append(data)
    
print('TRAIN:', len(train_dataset))

test_dataset = []
for data in dataset['test']:
    test_dataset.append(data)

print('TEST:', len(test_dataset))

val_dataset = []
for data in dataset['validation']:
    val_dataset.append(data)

print('VAL:', len(val_dataset))
        

save_data('conll2003_train.pkl', train_dataset)
save_data('conll2003_test.pkl', test_dataset)
save_data('conll2003_val.pkl', val_dataset)
