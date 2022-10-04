# coding: utf-8
import numpy as np

def tensor_to_str(tensor, tokenizer):
    tensor_ids = tensor.to('cpu').numpy()
    text = np.array(tokenizer.convert_ids_to_tokens(tensor_ids))
    return text

def untokenizing(tokenized_text, idx):
    """
    report = ""
    for ss in tokenized_text:
        if not ss.startswith('##'):
            report += ' ' + ss
        else:
            report += ss[2:]
    return report.strip()
    """
    l = 0
    text = []
    tmp_ = ""
    for i in idx:
      if i == True:
        if tmp_ == "":
          tmp_ = str(tokenized_text[l])
        else:
          tmp_ += ' ' + str(tokenized_text[l])
        l = l+1
      else:
        if tmp_ != "":
          text.append(tmp_.strip())
          tmp_ = ""
    if tmp_ != "":
      text.append(tmp_.strip())
    return text
          

def get_keyword(prediction, textlist, tag):
    prediction = np.array(prediction)
    textlist = np.array(textlist)
    idx = prediction==tag
    keyword_token = textlist[idx]
    keyword = untokenizing(keyword_token, idx)
    return keyword

def merging(tokenized_text, pred):
    tokens = []
    proba = []
    for i, ss in enumerate(tokenized_text):
        if not ss.startswith('##'):
            tokens.append(ss)
            proba.append(pred[i,:])
        else:
            tokens[-1] += ss[2:]
            proba[-1] += list(pred[i,:])
    proba = np.array(proba)
    return tokens, proba
