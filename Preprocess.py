# coding: utf-8
import re
import csv
import numpy as np
import tensorflow as tf


def pre_process_str(text):
    text = text.lower()
    text = re.sub("([^a-zA-Z0-9\u3131-\u3163\uac00-\ud7a3\n])+", " ", text)
    return text
	
def pre_process(reports_raw, keywords_raw):
    labels, labels_raw, in_abstract, not_nan_keyword = [], [], [], []
    text, text_raw, not_nan_text = [], [], []
    for i, keywords in enumerate(keywords_raw):
        report_pp = pre_process_str(reports_raw[i-1])#TODO -1 added
        token_s = report_pp.split()
        z = ['XX'] * len(token_s)
        split_keywords = [x.strip() for x in keywords.split(';')]
        #labels_raw.append(split_keywords)
        label_in_abstract = []
        for k in split_keywords:
            keyword = pre_process_str(k)
            if keyword == '': continue
            if keyword == ' ': continue
            if keyword == 'nan': continue
            keyword_list = keyword.split()
            if keyword in report_pp:
                idxl = len(keyword_list)
               # print( "keyword:",keyword,"report_pp",report_pp)
                for ii in range(len(token_s) -idxl +1 ):
                    #DEBUG
                    #print("ii:",ii,"idxl:",idxl,"token_s",token_s,len(token_s),"range:",len(token_s) - idxl + 1)
                    #print("keylist:", keyword_list[0], "ii:",ii, "token_s:",token_s[ii])
                    if keyword_list[0] in token_s[ii] :
                        tf = True
                        for jj in range(idxl):
                            if keyword_list[jj] not in token_s[ii+jj]:
                                tf = False
                        #print(tf)
                        if tf: break
                for jj in range(idxl):
                    token_s[ii + jj] = ['##']
                    z[ii + jj] = 'Keyword'
                label_in_abstract.append(keyword)
        if len(label_in_abstract)>0:
          text.append(report_pp)
          labels.append(z)
          in_abstract.append(label_in_abstract)
          not_nan_text.append(reports_raw[i])
          not_nan_keyword.append(split_keywords)
    return text, labels, not_nan_text, not_nan_keyword, in_abstract# reports_raw, labels_raw

def tokenizing(reports, label, tokenizer, MAX_LEN):
    tokenized_texts, corresponding_labels = [], []
    for i,s in enumerate(reports):
        tokenized_text, corresponding_label = [], []
        s_ = s.split()
        for j,ss in enumerate(s_):
            if j > MAX_LEN: break
            token=tokenizer.tokenize(ss)
            if len(token)==1:
                tokenized_text.append(token[0])
                corresponding_label.append(label[i][j])
            elif len(token)>1:
                 for token_ in token:
                    tokenized_text.append(token_)
                    corresponding_label.append(label[i][j])
        tokenized_texts.append(tokenized_text)
        corresponding_labels.append(corresponding_label)
    return tokenized_texts, corresponding_labels
