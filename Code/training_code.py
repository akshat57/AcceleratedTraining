import pandas as pd
import numpy as np
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import accuracy_score, classification_report
from load_data import initialize_data
from reading_datasets import read_ud_dataset
from labels_to_ids import tweebank_labels_to_ids
import time
import os
from useful_functions import load_data, save_data


def train(epoch, training_loader, model, optimizer, device, max_grad_norm = 10):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    
    for idx, batch in enumerate(training_loader):
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)

        #loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
        output = model(input_ids=ids, attention_mask=mask, labels=labels)
        tr_loss += output[0]

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
           
        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = output[1].view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        tr_labels.extend(labels)
        tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
    
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=max_grad_norm
        )
        
        # backward pass
        optimizer.zero_grad()
        output['loss'].backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    #print(f"Training loss epoch: {epoch_loss}")
    #print(f"Training accuracy epoch: {tr_accuracy}")

    return model


def testing(model, testing_loader, labels_to_ids, device):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    
    ids_to_labels = dict((v,k) for k,v in labels_to_ids.items())

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            
            #loss, eval_logits = model(input_ids=ids, attention_mask=mask, labels=labels)
            output = model(input_ids=ids, attention_mask=mask, labels=labels)

            eval_loss += output['loss'].item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
        
            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")
              
            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = output[1].view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
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
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    #print(f"Validation Loss: {eval_loss}")
    #print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions, eval_accuracy

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

def main(n_epochs, model_name, train_dataset_location, model_save_flag, model_save_location, model_load_flag, model_load_location, in_train_logfile):
    #Initialization training parameters
    max_len = 256
    train_batch_size = 32
    dev_batch_size = 32
    test_batch_size = 32
    learning_rate = 1e-05
    initialization_input = (max_len, train_batch_size, dev_batch_size, test_batch_size)

    #Reading datasets and initializing data loaders
    train_tb, dev_tb, test_tb, train_gum, dev_gum, test_gum, train_labels, dev_labels, test_labels = read_tb_gum()
    if train_dataset_location == 'GUM':
        train_data = train_gum
    else:
        train_data = load_data(train_dataset_location)
        

    input_data_gum = (train_data, dev_gum, test_gum, train_labels, dev_labels, test_labels)
    input_data_tb = (train_tb, dev_tb, test_tb, train_labels, dev_labels, test_labels)

    #Define tokenizer, model and optimizer
    device = 'cuda' if cuda.is_available() else 'cpu' #save the processing time
    if model_load_flag:
        tokenizer = AutoTokenizer.from_pretrained(model_load_location)
        model = AutoModelForTokenClassification.from_pretrained(model_load_location)
    else: 
        tokenizer =  AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(train_labels))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    model.to(device)

    #Get dataloaders
    train_loader, dev_loader, test_loader = initialize_data(tokenizer, initialization_input, input_data_gum)
    train_loader_tb, dev_loader_tb, test_loader_tb = initialize_data(tokenizer, initialization_input, input_data_tb)

    best_dev_acc = 0
    best_test_acc = 0
    best_epoch = -1
    best_tb_acc = 0
    best_tb_epoch = -1
    for epoch in range(n_epochs):
        start = time.time()
        print(f"Training epoch: {epoch + 1}")

        #train model
        model = train(epoch, train_loader, model, optimizer, device)
        
        #testing and logging
        labels_dev, predictions_dev, dev_accuracy = testing(model, dev_loader, dev_labels, device)
        print('DEV ACC:', dev_accuracy)
        
        labels_test, predictions_test, test_accuracy = testing(model, test_loader, test_labels, device)
        print('TEST ACC:', test_accuracy)
        
        labels_test_tb, predictions_test_tb, test_accuracy_tb = testing(model, test_loader_tb, test_labels, device)
        print('TB TEST ACC:', test_accuracy_tb)

        #saving model
        if dev_accuracy > best_dev_acc:
            best_dev_acc = dev_accuracy
            best_test_acc = test_accuracy
            best_epoch = epoch
            
            if model_save_flag:
                os.makedirs(model_save_location, exist_ok=True)
                tokenizer.save_pretrained(model_save_location)
                model.save_pretrained(model_save_location)

        if best_tb_acc < test_accuracy_tb:
            best_tb_acc = test_accuracy_tb
            best_tb_epoch = epoch

        #logging
        f = open(in_train_logfile, 'a')
        f.write('EPOCH: ' + str(epoch) + '\n\n')
        f.write('GUM TEST:' + '\n\n')
        f.write(classification_report(labels_test, predictions_test, digits = 5))
        f.write('\n\nTB TEST:' + '\n\n')
        f.write(classification_report(labels_test_tb, predictions_test_tb))
        f.write('\nDEV ACC : ' + str(round(dev_accuracy, 5)) + '\n')
        f.write('TEST ACC : ' + str(round(test_accuracy, 5)) + '\n')
        f.write('TB TEST ACC : ' + str(round(test_accuracy_tb, 5)) + '\n')
        f.write('BEST EPOCH : ' + str(best_epoch) + '\n')
        f.write('BEST ACCURACY --> ' +  'DEV:' +  str(round(best_dev_acc, 5)) + ', TEST:' + str(round(best_test_acc, 5)) + '\n')
        f.write('BEST TB TEST ACC : ' + str(round(best_tb_acc, 5)) + '\n')
        f.write('-'*80 + '\n')
        f.close()

        now = time.time()
        print('BEST ACCURACY --> ', 'DEV:', round(best_dev_acc, 5), 'TEST:',  round(best_test_acc, 5))
        print('TIME PER EPOCH:', (now-start)/60 )
        print()

    return best_dev_acc, best_test_acc, best_tb_acc, best_epoch, best_tb_epoch





if __name__ == '__main__':
    n_epochs = 15
    n_iterations = 5

    models = ['bert-base-uncased', 'roberta-base', 'xlm-roberta-base']
    train_dataset_directory = '../Datasets/POSTagging/GUM_augemented/'
    training_datasets = ['GUM']#only have strings in here. Also, remove .pkl


    for model_name in models:
        for dataset in training_datasets:
            if dataset == 'GUM':
                train_dataset_location = dataset
            else:
                train_dataset_location = train_dataset_directory  + dataset + '.pkl'

            #model saving parameters
            model_save_flag = True
            model_load_flag = False
            model_save_location = '../../saved_models/' + model_name + '_' + dataset
            model_load_location = None
    
            #logfile
            in_train_logfile = 'logs/training_logs/intrain_' + model_name + '_' + dataset + '.txt'
            result_logfile = 'logs/training_logs/results_' + model_name + '_' + dataset + '.txt'

            #initialize logfiles
            f = open(in_train_logfile, 'w')
            f.write('='*50 + '\n')
            f.write('MODEL NAME : ' + model_name + ' | ' + 'DATASET : ' + dataset + '\n')
            f.write('='*50 + '\n')
            f.close()

            g = open(result_logfile, 'w')
            g.write('='*50 + '\n')
            g.write('MODEL NAME : ' + model_name + ' | ' + 'DATASET : ' + dataset + '\n')
            g.write('='*50 + '\n')
            g.close()

            all_dev_acc, all_test_acc, all_test_tb_acc, all_best_epoch, all_best_tb_epoch = [], [], [], [], []
            for i in range(n_iterations):
                print(model_name, dataset, 'ITERATION:', i )

                f = open(in_train_logfile, 'a')
                f.write('ITERAION : ' + str(i) + '\n\n')
                f.close()

                best_dev_acc, best_test_acc, best_tb_acc, best_epoch, best_tb_epoch = main(n_epochs, model_name, train_dataset_location, model_save_flag, model_save_location, model_load_flag, model_load_location, in_train_logfile)
                all_dev_acc.append(best_dev_acc)
                all_test_acc.append(best_test_acc)
                all_test_tb_acc.append(best_tb_acc)
                all_best_epoch.append(best_epoch)
                all_best_tb_epoch.append(best_tb_epoch)

                #logging for results
                g = open(result_logfile, 'a')
                g.write('ITERAION : ' + str(i) + '\n')
                g.write('BEST DEV ACC : ' + str(round(best_dev_acc, 5)) + '\n')
                g.write('BEST TEST ACC : ' + str(round(best_test_acc, 5)) + '\n')
                g.write('BEST EPOCH : ' + str(best_epoch) + '\n')
                g.write('BEST TEST TB ACC : ' + str(round(best_tb_acc, 5)) + '\n')
                g.write('BEST TB EPOCH : ' + str(best_tb_epoch) + '\n')
                g.write('-'*30 + '\n')
                g.close()

            
            g = open(result_logfile, 'a')
            g.write('\nFINAL RESULTS : ' + '\n')
            g.write('MEAN DEV ACC : ' + str(round( np.mean(np.array(all_dev_acc)) * 100, 3)) + '|' + 'STD DEV ACC : ' + str(round( np.std(np.array(all_dev_acc)) * 100, 3)) + '\n')
            g.write('MEAN TEST ACC : ' + str(round( np.mean(np.array(all_test_acc)) * 100, 3)) + '|' + 'STD TEST ACC : ' + str(round( np.std(np.array(all_test_acc)) * 100, 3)) + '\n')
            g.write('MEAN BEST EPOCH : ' + str(round( np.mean(np.array(all_best_epoch)), 3)) + '\n')
            g.write('MEAN TES TB ACC : ' + str(round( np.mean(np.array(all_test_tb_acc)) * 100, 3)) + '|' + 'STD TEST ACC TB : ' + str(round( np.std(np.array(all_test_tb_acc)) * 100, 3)) +  '\n')
            g.write('MEAN BEST TBEPOCH : ' + str(round( np.mean(np.array(all_best_tb_epoch)), 3)) +  '\n')
            g.write('-'*30 + '\n')
            g.close()