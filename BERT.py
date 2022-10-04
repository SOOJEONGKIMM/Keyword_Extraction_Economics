from sklearn.linear_model import Ridge, Lasso

csv_list = ["QUARTERLY JOURNAL OF ECONOMICS","JOURNAL OF ECONOMIC PERSPECTIVES","ECONOMIC GEOGRAPHY",
            "JOURNAL OF FINANCE","JOURNAL OF ECONOMIC LITERATURE","Review of Environmental Economics and Policy",
            "JOURNAL OF FINANCIAL ECONOMICS","AMERICAN ECONOMIC REVIEW","JOURNAL OF POLITICAL ECONOMY","ENERGY ECONOMICS",
            "Journal of the Association of Environmental and Resource Economists","ENERGY POLICY",
            "REVIEW OF ECONOMIC STUDIES","American Economic Journal-Applied Economics","SMALL BUSINESS ECONOMICS",
            "JOURNAL OF ECONOMIC GROWTH","NBER Macroeconomics Annual","ECONOMIC POLICY",
            "Cambridge Journal of Regions Economy and Society","REVIEW OF ECONOMICS AND STATISTICS",
            "SOCIO-ECONOMIC PLANNING SCIENCES","ECOLOGICAL ECONOMICS"]

filename_read = csv_list
MAX_LEN = 256
batch_size_ = 16
learning_rate = 2e-5
epochs = 12

import csv
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification
from Preprocess import pre_process_str, pre_process, tokenizing
from Postprocess import tensor_to_str, untokenizing, get_keyword, merging

def flat_accuracy(preds, labels, masks):
    mask_flat = masks.flatten()
    pred_flat = np.argmax(preds, axis=2).flatten()[mask_flat==1]
    labels_flat = labels.flatten()[mask_flat==1]
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


reports_raw = []
keywords_raw = []
# Read Data
start = time.time()

document_type = ['Article', 'Article; Early Access', 'Article; Proceedings Paper', 'Article; Proceedings Paper; Retracted Publication',  'Review']

print("Start processing..  ")
for i in csv_list:
    reader = pd.read_csv(i + '.csv', header=0)
    #article = reader[reader['Document Type'] in document_type]
    #article = reader[reader['Document Type'].isin(document_type)]
    article = reader[reader['DT'].isin(document_type)]
    #article.Abstract = article.Abstract.astype(str)
    article.AB = article.AB.astype(str)#Abstract

    #combine author and keywords
    #article['Combined_Keywords'] = article['Author Keywords'].astype(str)+';'+article['Keywords Plus'].astype(str)
    article['Combined_Keywords'] = article['DE'].astype(str) + ';' + article['ID'].astype(str)
    #article.dropna(subset=['Abstract','Combined_Keywords'],inplace=True)
    #reports_raw.extend(article['Abstract'])
    #dropna() removes missing values
    article.dropna(subset=['AB', 'Combined_Keywords'], inplace=True)
    reports_raw.extend(article['AB'])
    #print("reports_raw:",reports_raw)
    keywords_raw.extend(article['Combined_Keywords'])
print("done reading csv.", time.time()-start)
print("total elements, ", len(reports_raw))

# Pre-processing
#reports, labels, text_raw, labels_raw, in_abstract = pre_process(reports_raw, keywords_raw)
reports, labels, not_nan_text, real_keywords, in_abstract = pre_process(reports_raw, keywords_raw)
print("done preprocessing.", time.time() - start)
#print("reports:",reports[0])
print("total articles without NaN, ", len(reports))
#print("non nan text",not_nan_text)
# Labels
lab2idx = {'XX': 0, 'Keyword': 1}

# Word-piece tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenizing
tokenized_texts, corresponding_labels = tokenizing(reports, labels, tokenizer, MAX_LEN)
print("correspoding:",corresponding_labels)
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
labs = pad_sequences([[lab2idx.get(l) for l in lab] for lab in corresponding_labels],
                     maxlen=MAX_LEN, value=lab2idx["XX"], padding="post",
                     dtype="long", truncating="post")
attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]
print("done tokenizing.", time.time() - start)
print("input_ids:",len(input_ids))
print("labs:",len(labs))
print("attention_masks:",len(attention_masks))
print("in_abstract:",len(in_abstract))
print("attention_masks:",len(not_nan_text))
print("in_abstract:",len(real_keywords))

# Split training/test sets
#modified from random_state=12345, test_size=0.1
tr_inputs, val_inputs, tr_labs, val_labs = train_test_split(input_ids, labs, random_state=789, test_size=0.5)
tr_masks, val_masks, tr_abstract, val_abstract = train_test_split(attention_masks, in_abstract, random_state=789, test_size=0.5)
tr_text, val_text, tr_keywords, val_keywords = train_test_split(not_nan_text, real_keywords, random_state=789, test_size=0.5)
#tr_text, val_text, tr_keywords, val_keywords = train_test_split([x for x in text_raw], [x for x in labels_raw],
#                                                                random_state=12345, test_size=0.1)
print("val_inputs:",val_inputs)
print("val_abstract:",val_abstract)
# Unique datasets (겹치지않는 고유한 요소 배열)
nptr_keywords = np.array([[pre_process_str(x[0])] for x in tr_keywords])
npts_keywords = np.array([[pre_process_str(x[0])] for x in val_keywords])
print("Training Unique: {}".format(np.unique(nptr_keywords[:, 0]).shape[0]))
print("Test Unique: {}".format(np.unique(npts_keywords[:, 0]).shape[0]))
print("Intersect of Train and Test keywords: {}".format(np.intersect1d(nptr_keywords[:, 0], npts_keywords[:, 0]).shape[0]))

csv_save_1 = open('Count_train.csv', 'w', newline='')
csvwriter_1 = csv.writer(csv_save_1)
csvwriter_1.writerow(['Keywords', 'Count'])
uniq_1 = np.asarray(np.unique(nptr_keywords[:, 0], return_counts=True)).T
for i in range(uniq_1.shape[0]):
    csvwriter_1.writerow([uniq_1[i][0], uniq_1[i][1]])
csv_save_1.close()


csv_save_2 = open('Count_test.csv', 'w', newline='')
csvwriter_2 = csv.writer(csv_save_2)
csvwriter_2.writerow(['Keywords', 'Count'])
uniq_2 = np.asarray(np.unique(npts_keywords[:, 0], return_counts=True)).T
for i in range(uniq_2.shape[0]):
    csvwriter_2.writerow([uniq_2[i][0], uniq_2[i][1]])
csv_save_2.close()


# Load Data
# torch.Tensor is a multi-dimensional matrix containing elements of a single data type
tr_inputs = torch.tensor(tr_inputs).to(torch.long)
val_inputs = torch.tensor(val_inputs).to(torch.long)
tr_labs = torch.tensor(tr_labs).to(torch.long)
val_labs = torch.tensor(val_labs).to(torch.long)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)
train_data = TensorDataset(tr_inputs, tr_masks, tr_labs)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size_)
valid_data = TensorDataset(val_inputs, val_masks, val_labs)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size_)

# model
#model = BertForTokenClassification.from_pretrained(bert_model_dir, num_labels=self.opt.tag_nums)
#bert_model_dir: bert pre-training model parameters
#num_labels: The number of word tag classes. which is(2 or 3)*type+1
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(lab2idx))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.cuda() if device.type == 'cuda' else model.cpu()

# Fine-tuning parameters
FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}

    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)

train_loss, valid_loss = [], []
valid_accuracy = []
# Exact Matching
EM = []
max_grad_norm = 1.0

for iter, _ in tqdm(enumerate(range(epochs))):
    # training
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        '''
        random_src = RandomizedSearchCV(estimator=model, param_distributions=model.parameters(), cv=2, n_iter=10, n_jobs=-1)
        model.fit(tr_inputs, val_inputs)
        model.fit(tr_labs, val_labs)
        model.fit(tr_masks, val_masks)
        model.fit(tr_abstract, val_abstract)
        model.fit(tr_text, val_text)
        model.fit(tr_keywords, val_keywords)
        print("best score of random search", random_src.best_score_)
        print("best param of random search", random_src.best_params_)
        print("best estimator of random search", random_src.best_estimator_)
        '''
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss.backward()
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        model.zero_grad()
    print("Train loss: {}".format(tr_loss / nb_tr_steps))
    train_loss.append(tr_loss / nb_tr_steps)
    # evaluation
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0
    predictions, true_labels = [], []
    matching = []
    for ii, batch in enumerate(valid_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        masks = b_input_mask.to('cpu').numpy()
        for jj, _ in enumerate(label_ids):
            pred = logits[jj]
            mask = masks[jj]
            if np.sum(mask) == 0: continue
            pred_lab = np.argmax(pred, axis=1)
            pred_lab_ = pred_lab[mask == 1]
            #predictions.append(pred_lab_)
            true_lab = val_labs[ii * batch_size_ + jj].to('cpu').numpy()
            true_lab_ = true_lab[mask == 1]
            true_labels.append(true_lab_)
            token_text = tensor_to_str(b_input_ids[jj], tokenizer)
            tkns = token_text[mask == 1]
            pred_token, mpred = merging(tkns, pred)
            pred_label = np.argmax(mpred, axis=1)
            specimen = get_keyword(pred_label, pred_token, 1)
            true_specimen = get_keyword(true_lab, token_text, 1)
            matching.append(int(specimen == true_specimen))
            predictions.append(specimen)

        tmp_eval_accuracy = flat_accuracy(logits, label_ids, masks)
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    valid_loss.append(eval_loss)
    valid_accuracy.append(eval_accuracy / nb_eval_steps)
    matching = np.array(matching)
    EM.append(np.average(matching))
    print("Loss: {}".format(eval_loss))
    print("Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    print("Exact Matching: {}".format(np.average(matching)))


def intersect(a, b):
  return list(set(a) & set(b))

csv_save = open('Output_4.csv', 'w', newline='')
csvwriter = csv.writer(csv_save)
csvwriter.writerow(['Inputs', 'True Keywords','True Keywords in Abstract', 'Predictions', 'Precision', 'Recall'])
for i in range(len(val_inputs)):
    if len(predictions[i]) == 0: precision_ = 0
    else : precision_ = len(intersect(val_abstract[i], predictions[i]))/len(predictions[i])
    recall_ = len(intersect(val_abstract[i], predictions[i]))/len(val_abstract[i])
    #F1_ = 2 * precision_ * recall_ /(precision_ + recall_)
    csvwriter.writerow([val_text[i], val_keywords[i], val_abstract[i], predictions[i], precision_, recall_])
csv_save.close()
