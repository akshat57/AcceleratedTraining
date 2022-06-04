import pandas as pd
import numpy as np
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import accuracy_score, classification_report
from seqeval.metrics import f1_score
from load_data import initialize_data
from reading_datasets import read_ud_dataset
from labels_to_ids import tweebank_labels_to_ids
import time
import os
from useful_functions import load_data, save_data


def testing(model, testing_loader, labels_to_ids):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    collect_predictions = []
    collected_predictions = []
    
    ids_to_labels = dict((v,k) for k,v in labels_to_ids.items())

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            
            #loss, eval_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
            output = model(input_ids=ids, attention_mask=mask, labels=labels)

            eval_loss += output[0].item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
        
            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")
              
            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = output[1].view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            #get predictions
            batch_predictions = torch.argmax(output[1], axis=2)
            for i in range(batch_predictions.shape[0]):
                pred_seq = batch_predictions[i]
                label_seq = labels[i]
                collect_predictions.append(pred_seq[label_seq != -100]) 

            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.extend(labels)
            eval_preds.extend(predictions)
            
            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [ids_to_labels[id.item()] for id in eval_labels]
    predictions = [ids_to_labels[id.item()] for id in eval_preds]

    for seq in collect_predictions:
         collected_predictions.append([ids_to_labels[id.item()] for id in seq] )
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    #print(f"Validation Loss: {eval_loss}")
    #print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions, eval_accuracy, collected_predictions
    
def read_tb_gum():
    tb_location = '../Datasets/POSTagging/Tweebank/'
    train_tb = read_ud_dataset(dataset = 'tb', location = tb_location, split = 'train')
    dev_tb = read_ud_dataset(dataset = 'tb', location = tb_location, split = 'dev')
    test_tb = read_ud_dataset(dataset = 'tb', location = tb_location, split = 'test')

    gum_location = '../Datasets/POSTagging/GUM/'
    train_gum = read_ud_dataset(dataset = 'gum', location = gum_location, split = 'train')
    dev_gum = read_ud_dataset(dataset = 'gum', location = gum_location, split = 'dev')
    test_gum = read_ud_dataset(dataset = 'gum', location = gum_location, split = 'test')

    train_labels = tweebank_labels_to_ids
    dev_labels = tweebank_labels_to_ids
    test_labels = tweebank_labels_to_ids

    return train_tb, dev_tb, test_tb, train_gum, dev_gum, test_gum, train_labels, dev_labels, test_labels


if __name__ == '__main__':
    #Initialization parameters
    max_len = 256
    train_batch_size = 32
    dev_batch_size = 32
    test_batch_size = 32
    learning_rate = 1e-05
    initialization_input = (max_len, train_batch_size, dev_batch_size, test_batch_size)
    epochs = 15
    save_directory = '../../saved_models/gum_aug_sym'

    #Reading datasets and initializing data loaders
    train_tb, dev_tb, test_tb, train_gum, dev_gum, test_gum, train_labels, dev_labels, test_labels = read_tb_gum()
    input_data_gum = (train_gum, dev_gum, test_gum, train_labels, dev_labels, test_labels)
    input_data_tb = (train_tb, dev_tb, test_tb, train_labels, dev_labels, test_labels)

    #Define tokenizer, model and optimizer
    device = 'cuda' if cuda.is_available() else 'cpu' #save the processing time
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
    model = AutoModelForTokenClassification.from_pretrained(save_directory)    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    model.to(device)

    #Get dataloaders
    train_loader_tb, dev_loader_tb, test_loader_tb = initialize_data(tokenizer, initialization_input, input_data_tb)
    #train_loader, dev_loader, test_loader = initialize_data(tokenizer, initialization_input, input_data_gum)

    labels_test_tb, predictions_test_tb, test_accuracy_tb, collect_predictions = testing(model, test_loader_tb, test_labels)

    print(classification_report(labels_test_tb, predictions_test_tb, digits = 5))


    save_data('logs/tb_predictions.pkl', collect_predictions)