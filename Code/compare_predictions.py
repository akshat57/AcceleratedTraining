from analyse_data import read_tb_gum
from useful_functions import load_data

train_tb, dev_tb, test_tb, train_gum, dev_gum, test_gum = read_tb_gum()
predictions_tb = load_data('logs/tb_predictions.pkl')

for (tokens, labels), predictions in zip(test_tb, predictions_tb):
    for token, label, pred in zip(tokens, labels, predictions):
        if label == 'NOUN' and label != pred:
            print(token, pred, '|', ' '.join(tokens))