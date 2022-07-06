"""
First author: HongSeok Choi (hongking9@gist.ac.kr)
Corresponding author: Hyunju Lee (hyunjulee@gist.ac.kr)
Code for Early Stopping Based on Unlabeled Samples in Text Classification.
"""
import os
import shutil
import preprocessing as pp
from sklearn.utils import shuffle

task = "SST-2"
task_path = os.path.join("./data",task,"full_data")
rand_seed = 12345
K = 50
num_test = 5
exist_ok = True

def write_tsv(write_path, examples):
    f = open(write_path,'w')
    f.write('label\ttext\n')
    for ex in examples:
        f.write(ex.label+'\t'+ex.text_a+'\n')
    f.close()

def write_unl_tsv(write_path, examples):
    f = open(write_path,'w')
    f.write('text\n')
    for ex in examples:
        f.write(ex.text_a+'\n')
    f.close()

task = task.strip()
Processor = pp.task_to_processor(task)
processor = Processor(task, task_path, tokenizer=None, max_length=64) 

label_list = processor.get_label_list()
lab_examples = processor.tsv_to_examples('labeled.tsv')
tst_examples = processor.tsv_to_examples('test_with_gold.tsv') 
unl_examples = processor.tsv_to_examples('unlabeled.tsv')

examples_per_class = {label:[] for label in label_list}
for ex in shuffle(lab_examples, random_state=rand_seed):
    examples_per_class[ex.label].append(ex)

'''For balanced setting, we sample 50 samples per class.
The class distribution of the test set is already balanced.'''
for test_i in range(num_test):
    write_path = os.path.join("./data",task,"bal",str(test_i))
    sampled_examples = []
    for label in label_list:
        sampled_examples += examples_per_class[label][K*test_i:K*(test_i+1)]
    
    os.makedirs(write_path, exist_ok=exist_ok)
    write_tsv(os.path.join(write_path,'labeled.tsv'), sampled_examples)
    shutil.copyfile(os.path.join(task_path,'test_with_gold.tsv'), os.path.join(write_path,'test_with_gold.tsv'))
    shutil.copyfile(os.path.join(task_path,'test.tsv'), os.path.join(write_path,'test.tsv'))
    shutil.copyfile(os.path.join(task_path,'unlabeled.tsv'), os.path.join(write_path,'unlabeled.tsv'))
    
'''In imbalanced setting, the class distribution 
of the test set is adjusted to 2:8 (neg:pos).'''
tst_examples_per_class = {label:[] for label in label_list}
for ex in shuffle(tst_examples, random_state=rand_seed):
    tst_examples_per_class[ex.label].append(ex)

for test_i in range(num_test):
    write_path = os.path.join("./data",task,"imbal",str(test_i))
    sampled_examples = []
    for label in label_list:
        sampled_examples += examples_per_class[label][K*test_i:K*(test_i+1)]
    sampled_tst_examples = tst_examples_per_class['0'][:200] + tst_examples_per_class['1'][:800]
    
    os.makedirs(write_path, exist_ok=exist_ok)
    write_tsv(os.path.join(write_path,'labeled.tsv'), sampled_examples)
    write_tsv(os.path.join(write_path,'test_with_gold.tsv'), sampled_tst_examples)
    write_unl_tsv(os.path.join(write_path,'test.tsv'), sampled_tst_examples)
    shutil.copyfile(os.path.join(write_path,'test.tsv'), os.path.join(write_path,'unlabeled.tsv'))
    
print('done')