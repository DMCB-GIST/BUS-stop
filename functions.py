# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 03:25:43 2021

@author: Anonymous

functions
"""

import pickle
import random 
import numpy as np

from tensorflow.python.client import device_lib
from sklearn.metrics import accuracy_score
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from sklearn.metrics import f1_score

import nltk
nltk.download('punkt')

#print(device_lib.list_local_devices())
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def dump_all(all_data, data_path):
    with open(data_path, 'wb') as handle:
        pickle.dump(all_data, handle)

def load_all(data_path):
    with open(data_path, 'rb') as handle:
        load_all = pickle.loads(handle.read())
    return load_all

class InputExample(object):
  def __init__(self, guid, text_a, text_b=None, label=None):
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

class InputFeatures(object):
  def __init__(self, input_ids, input_mask, segment_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id # also can be numerical values
    
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i
  
  features = []
  for (ex_index, example) in enumerate(examples):
    
    if max_seq_length<=64:
      tokens_a = tokenizer.tokenize(example.text_a)
    else:
      sentences = sent_tokenize(example.text_a)
      tokens_set = []
      for ti in range(len(sentences)):
        tokens_set.append(tokenizer.tokenize(sentences[ti]))
        
      tokens_a = []
      for ti in range(len(tokens_set)-1):
        tokens_a += tokens_set[ti]+['[SEP]']
      tokens_a += tokens_set[-1]

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
      _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
      if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    if example.label=='-':
      label_id = -1
    else:
      label_id = label_map[example.label]
    '''
    if ex_index < 5:
      tf.logging.info("*** Example ***")
      tf.logging.info("guid: %s" % (example.guid))
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in tokens]))
      tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.logging.info(
          "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
      #tf.logging.info("%s , %s" % (example.label, label_id))
    '''
    features.append(
        InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id))
  return features

class TensorInputs(object):
  def __init__(self, input_ids, input_mask, segment_ids, label_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
  def shuffles(self,seed=0,cut=None):
    data_len = len(self.label_ids)
    data_idxs = np.random.RandomState(seed=seed).permutation(data_len)
    data_input_ids = []
    data_input_mask = []
    data_segment_ids = []
    data_label_ids = []
    for idx in data_idxs[:cut]:
        data_input_ids.append(self.input_ids[idx])
        data_input_mask.append(self.input_mask[idx])
        data_segment_ids.append(self.segment_ids[idx])
        data_label_ids.append(self.label_ids[idx])
    rtrn = TensorInputs(data_input_ids,data_input_mask,
                        data_segment_ids,data_label_ids)
    return rtrn

class my_logging(object):
  def __init__(self, data_dir):
    self.data_dir = data_dir
    with open(self.data_dir,'w') as f:
      f.write('==Logging Start==\n')
  def printNlogging(self,*lst):
    lst = list(lst)
    strs = ''
    for elem in lst:
        strs += str(elem)+' '
    print(strs)
    with open(self.data_dir,'a') as f:
      f.write(strs+'\n')
    return strs+'\n'

def buildTensorInputs(features):
    input_ids,input_mask,segment_ids,label_ids = [], [], [], []
    for feature in features:
        input_ids.append(feature.input_ids)
        input_mask.append(feature.input_mask)
        segment_ids.append(feature.segment_ids)
        label_ids.append(feature.label_id)
    data = TensorInputs(input_ids,input_mask,segment_ids,label_ids)
    return data

def str2exec(bi,btchs,data="trn",lr=0.0,dr=0.0):
    bi = str(bi)
    btchs = str(btchs)
    str_code = ""
    common_str = "["+bi+"*"+btchs+":("+bi+"+1)*"+btchs+"],"
    str_code+= "{input_ids:"+data+".input_ids"+common_str
    str_code+= " input_mask:"+data+".input_mask"+common_str
    str_code+= " segment_ids:"+data+".segment_ids"+common_str
    str_code+= " labels:"+data+".label_ids"+common_str
    str_code+= " learn_intf:"+str(lr)+","
    str_code+= " drop_rate:"+str(dr)+",}"
    return str_code

def round_multiples(lst):
    rlst = []
    for elem in lst:
        rlst.append(round(elem,4))
    return rlst

def shuffle_list(samples,seed=0,rand_indices=None):
    if type(rand_indices)==type(None):
        rand_indices = np.random.RandomState(seed=seed).permutation(len(samples))
    shuffled = []
    for i in rand_indices:
        shuffled.append(samples[i])
    return shuffled

def text_reader(txt_path):
    samples = []
    f = open(txt_path,'r')
    for line in f.readlines():
        sent,label = line.strip().split('\t')
        samples.append((label,sent))
    f.close()
    return samples

def select_by_index(lst,indices,reverse=False):
    selected = []
    if reverse:
        for i in range(len(lst)):
            if i in indices:
                continue
            selected.append(lst[i])
    else:
        for i in indices:
            selected.append(lst[i])
    return selected

def pick_samples_per_class(examples,label_list,si,pick):
    samples_per_class = defaultdict(list)
    for example in examples:
        samples_per_class[example.label].append(example)
    
    samples = []
    for label in label_list:
        label_len = len(samples_per_class[label])
        indices = [i%label_len for i in range(si*pick,(si+1)*pick)]
        samples += select_by_index(samples_per_class[label],indices)
    
    return samples

def samples2examples(samples):
    examples = []
    for sample_i,sample in enumerate(samples):
        guid = 'data-'+str(sample_i)
        example = InputExample(guid=guid, text_a=sample[1], text_b='', label=sample[0])
        examples.append(example)
    return examples

def tsv2examples(tsv_path):
    examples = []
    f = open(tsv_path,'r',errors='ignore')
    for line in f.readlines():
        guid = 'data-'+str(len(examples))
        label,text = line.strip().split('\t')
        example = InputExample(guid=guid, text_a=text, text_b='', label=label)
        examples.append(example)
    f.close()
    return examples

def build_trn_set(trn_org,epochs):
    in_ids,in_mask,seg_ids,lab_ids = [], [], [], []
    for gather in range(epochs+1):
        #trn_org = buildTrain(trn_features)
        trn = trn_org.shuffles(seed=np.random.randint(999))
        in_ids += trn.input_ids
        in_mask += trn.input_mask
        seg_ids += trn.segment_ids
        lab_ids += trn.label_ids
    trn = TensorInputs(in_ids,in_mask,seg_ids,lab_ids)
    #trn_bnum = int(np.ceil(len(trn.label_ids)/btchs))
    return trn


def pick_idx_per_class(examples,label_list,seed,pick):
    indices_per_class = defaultdict(list)
    for i,example in enumerate(examples):
        indices_per_class[example.label].append(i)
    
    #sample_indices = []
    sample_indices = {}
    for label in label_list:
        label_len = len(indices_per_class[label])
        class_indices = np.arange(label_len)
        random.Random(seed).shuffle(class_indices)
        class_indices = class_indices[:pick]
        #sample_indices += select_by_index(indices_per_class[label],class_indices)
        sample_indices[label] = select_by_index(indices_per_class[label],class_indices)
    return sample_indices

def pick_samples_from_idx(indices,examples,reverse=False):
    sel_examples = []
    if reverse==False:
        for i in indices:
            sel_examples.append(examples[i])
    else:
        indices = set([i for i in range(len(examples))]) - set(indices)
        indices = sorted(indices)
        for i in indices:
            sel_examples.append(examples[i])
    return sel_examples

def evenly_shuffle(lst_of_lst):
    idices_mark = []
    all_lst = []
    for lst in lst_of_lst:
        all_lst += lst
        len_lst = len(lst)
        idices_mark += [i/len_lst for i in range(len_lst)]
    
    evenly_lst = []
    for i in np.argsort(idices_mark):
        evenly_lst.append(all_lst[i])
    
    return evenly_lst

def calc_hardness(probs_set):
    T = np.shape(probs_set)[0]
    num_labels = np.shape(probs_set)[2]
    sump_c = 0.0
    sumplogp = 0.0
    for ci in range(num_labels):
        sump = 0.0
        for di in range(T):
            sump += probs_set[di,:,ci]
            sumplogp += probs_set[di,:,ci]*np.log(probs_set[di,:,ci])
        sump /= T
        sump_c += sump*np.log(sump)
    sumplogp /= T
    sump_c = -sump_c
    hardness = sumplogp + sump_c
    return hardness

def probs2dist(data_probs):
    num_labels = data_probs.shape[-1]
    data_dist = num_labels*[0.]
    for label_id in data_probs.argmax(1):
        data_dist[label_id] += 1.
    data_dist = np.array(data_dist)/np.sum(data_dist)
    return data_dist

def cross_entropy(probs,label_ids):
    loss = 0.0
    for i in range(len(label_ids)):
        loss += np.log(probs[i][label_ids[i]])
    return -loss/len(label_ids)

def calc_ece_and_oe(prob_per_bin,crrt_per_bin):
    ece = 0.0
    oe = 0.0
    data_size = 0
    for bi in range(len(prob_per_bin)):
        size_bm = len(prob_per_bin[bi])
        data_size += size_bm
        if size_bm==0:
            continue
        conf_bm = np.mean(prob_per_bin[bi])
        acc_bm = np.mean(crrt_per_bin[bi])
        ece += size_bm*abs(acc_bm-conf_bm)
        oe += size_bm*(conf_bm*max(conf_bm-acc_bm,0.0))
    ece /= data_size
    oe /= data_size
    return ece, oe

def rounds(lst,cut=4):
    return list(np.around(np.array(lst),cut))

def data2label_list(data):
    if data=='AGNews':
        return ['1','2','3','4']
    if data=='DBPedia':
        return ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
    if data=='Elec':
        return ['1','2']
    if data=='IMDB':
        return ['0','1']
    if 'SST-2' in data:
        return ['0','1']
    return []

def EUD_mat(vecs):
    num_samples = len(vecs)
    eud_mat = np.zeros((num_samples,num_samples),'float32')
    for i1 in range(num_samples):
        for i2 in range(i1+1,num_samples):
            eud = np.linalg.norm(vecs[i1]-vecs[i2])
            eud_mat[i1][i2] = eud
            eud_mat[i2][i1] = eud
    return eud_mat

def LID(eud_mat,m=20):
    num_samples = len(eud_mat)
    lid = 0.0
    for xi in range(num_samples):
        nears_to_m = np.argsort(eud_mat[xi])[:m+1]
        rm_xi = eud_mat[xi][nears_to_m[-1]]
        lid_of_xi = 0.0
        for mi in nears_to_m:
            if xi==mi:
                continue
            ri_xi = eud_mat[xi][mi]
            lid_of_xi += np.log((ri_xi+1e-9)/(rm_xi+1e-9))
        lid_of_xi = m/(lid_of_xi+1e-9)
        lid += lid_of_xi
    lid = -lid/num_samples
    return lid

def label_counter(examples,label_list):
    count_dict = {label:0 for label in label_list}
    for example in examples:
        count_dict[example.label] += 1
    return count_dict
        