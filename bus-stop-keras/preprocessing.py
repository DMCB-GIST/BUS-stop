"""
First author: HongSeok Choi (hongking9@gist.ac.kr)
Corresponding author: Hyunju Lee (hyunjulee@gist.ac.kr)
Code for Early Stopping Based on Unlabeled Samples in Text Classification.
"""
import os
import logging
import numpy as np

logger = logging.getLogger('BUS-stop')

def task_to_label_list(task):
    key = task.lower().strip()
    if key=='sst-2':
        return ['0','1']
    if key=='mrpc':
        return ['0','1']
    '''You can add another tasks'''
    raise KeyError("There is no label list for task '{}'".format(task))

def task_to_processor(task):
    key = task.lower().strip()
    if key=='sst-2':
        return TEXT1_processor
    if key=='mrpc':
        return TEXT2_processor
    '''You can add another tasks'''
    raise KeyError("There is no processor for task '{}'. Check the exact task name.".format(task))

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def tsv_to_examples_for_text1or2_format(tsv_path, label_list, do_pairwise=False):
    f = open(tsv_path,'r',errors='ignore')
    lines = f.readlines()
    column_names = lines[0].strip().split('\t')
    if not do_pairwise:
        assert column_names==['label','text'] or column_names==['text'], \
            "The tsv format should be one of ['text', 'label'\\tab'text']"
    else:
        assert column_names==['label','text1','text2'] or column_names==['text1','text2'], \
            "The tsv format should be one of ['text1'\\tab'text2', 'label'\\tab'text1'\\tab'text2']"
    num_columns = len(column_names)
    
    label_count = {label:0 for label in label_list}
    label_count[None] = 0
    
    examples = []
    for line_i,line in enumerate(lines[1:]):
        line_i += 1

        guid = 'data-'+str(line_i)
        attributes = line.strip().split('\t')
        assert len(attributes)==num_columns, \
            "Row {} in this tsv file does not match format.".format(line_i+1)

        label,text_a,text_b = None,'',''
        if not do_pairwise:
            if len(attributes)==1:
                text_a = attributes[0]
            else: #len(attributes)==2:
                label,text_a = attributes
        else:
            if len(attributes)==2:
                text_a,text_b = attributes
            else: #len(attributes)==3:
                label,text_a,text_b = attributes
        
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        label_count[label] += 1
        
    logger.info("In file {}, we read {} samples, ".format(tsv_path,len(examples)))
    logger.info("where the class distribution is {}.".format(label_count))
    f.close()
    
    return examples

class TEXT1_processor(object):
    def __init__(self, task, task_path, tokenizer, max_length):
        self.task = task
        self.task_path = task_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_list = task_to_label_list(self.task)
        label_map = {}
        for (i, label) in enumerate(self.label_list):
            label_map[label] = i
        label_map[None] = -1 # for unlabeled samples
        self.label_map = label_map
    def get_label_list(self):
        return self.label_list
    def tsv_to_examples(self,tsv_name):
        tsv_path = os.path.join(self.task_path, tsv_name)
        return tsv_to_examples_for_text1or2_format(tsv_path, 
                                                   self.label_list)
    def examples_to_features(self,examples):
        text1_all = [ex.text_a for ex in examples]
        X =  self.tokenizer(text1_all, padding='max_length', #True or 'max_length'
                            truncation=True, max_length=self.max_length)
        y = [self.label_map[ex.label] for ex in examples]
        if "token_type_ids" not in X:
            token_type_ids = np.zeros((len(X["input_ids"]), self.max_length), dtype=int)
        else:
            token_type_ids = np.array(X["token_type_ids"])
        
        return {"input_ids": np.array(X["input_ids"]), "token_type_ids": token_type_ids, 
                "attention_mask": np.array(X["attention_mask"])}, np.array(y)

class TEXT2_processor(object):
    def __init__(self, task, task_path, tokenizer, max_length):
        self.task = task
        self.task_path = task_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_list = task_to_label_list(self.task)
        label_map = {}
        for (i, label) in enumerate(self.label_list):
            label_map[label] = i
        label_map[None] = -1 # for unlabeled samples
        self.label_map = label_map
    def get_label_list(self):
        return self.label_list
    def tsv_to_examples(self,tsv_name):
        tsv_path = os.path.join(self.task_path, tsv_name)
        return tsv_to_examples_for_text1or2_format(tsv_path, self.label_list, 
                                                   do_pairwise=True)
    def examples_to_features(self,examples):
        text1_all = [ex.text_a for ex in examples]
        text2_all = [ex.text_a for ex in examples]
        X =  self.tokenizer(text1_all, text2_all, padding='max_length', 
                            truncation=True, max_length=self.max_length)
        y = [self.label_map[ex.label] for ex in examples]
        if "token_type_ids" not in X:
            token_type_ids = np.zeros((len(X["input_ids"]), self.max_length))
        else:
            token_type_ids = np.array(X["token_type_ids"])
        return {"input_ids": np.array(X["input_ids"]), "token_type_ids": token_type_ids, 
                "attention_mask": np.array(X["attention_mask"])}, np.array(y)

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

def select_by_index(lst,indices,reverse=False):
    selected = []
    if not reverse:
        for i in indices:
            selected.append(lst[i])
    else:
        for i in range(len(lst)):
            if i in indices:
                continue
            selected.append(lst[i])
    if isinstance(lst, np.ndarray):
        selected = np.array(selected)
    return selected
