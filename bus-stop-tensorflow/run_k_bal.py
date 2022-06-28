# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 14:01:00 2021

@author: Anonymous
"""

import os
import argparse
import tensorflow as tf
import modeling
import tokenization
import numpy as np
import time
import functions as fs
import random

from scipy.stats import entropy
from collections import Counter,deque
from sklearn.metrics import pairwise_distances

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu',default='0',required=True)
arg_parser.add_argument('--data',required=True)
arg_parser.add_argument('--k',default=50,required=False)
#arg_parser.add_argument('--valid_k',default=25,required=False)
arg_parser.add_argument('--valid',default=0.25,required=False)
arg_parser.add_argument('--epochs',default=15,required=False)
arg_parser.add_argument('--start',default=0,required=False)
arg_parser.add_argument('--end',default=10,required=False)
arg_saved = arg_parser.parse_args()

gpu = str(arg_saved.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
data = str(arg_saved.data)
k_shot = int(arg_saved.k)
valid_k = int(k_shot*float(arg_saved.valid))
if data=='DBPedia':
    max_seq_length = 128
elif data=='Elec':
    max_seq_length = 256
elif data=='IMDB':
    max_seq_length = 512
else:
    max_seq_length = 64
start = int(arg_saved.start)
end = int(arg_saved.end)
epochs = int(arg_saved.epochs)

"""
gpu = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
data = str('SST-2')
k_shot = int(50)
max_seq_length = 64
start = int(0)
end = int(10)
epochs = 30
"""

def calc_calibration(data_prob,label_ids,bins=10):
    data_len = len(label_ids) #
    base_prob = 1.0/len(label_list)
    unit_prob = (1.0-base_prob)/bins
    
    prob_per_bin = {bi:[] for bi in range(bins)}
    crrt_per_bin = {bi:[] for bi in range(bins)}
    for di in range(data_len):
        pred_i = np.argmax(data_prob[di])
        pred_prob = data_prob[di][pred_i]
        true_i = label_ids[di]
        iscorrect = float(true_i == pred_i)
        for bi in range(bins):
            if pred_prob > bi*unit_prob+base_prob and pred_prob <= (bi+1)*unit_prob+base_prob:
                prob_per_bin[bi].append(pred_prob)
                crrt_per_bin[bi].append(iscorrect)
                break
    return prob_per_bin,crrt_per_bin

def eval_prob_set(data_str,dr=0.0):
    label_ids = eval(data_str).label_ids
    data_bnum = int(np.ceil(len(label_ids)/(btchs*20)))
    data_probs = []
    for bi in range(data_bnum):
        data_bprob = sess.run(probs, 
                     feed_dict=eval(fs.str2exec(bi,20*btchs,data_str,dr=dr)))
        data_probs += list(data_bprob)
    return np.array(data_probs)

def eval_acc_loss(data_str):
    data_probs = eval_prob_set(data_str)
    data_acc,data_loss,_,_ = eval_all_metrics(data_probs,data_str,bins=0)
    return data_acc,data_loss

def eval_vec_set(data_str,dr=0.0):
    label_ids = eval(data_str).label_ids
    data_bnum = int(np.ceil(len(label_ids)/(btchs*20)))
    data_vecs = []
    for bi in range(data_bnum):
        data_bvec = sess.run(output_layer, 
                     feed_dict=eval(fs.str2exec(bi,20*btchs,data_str,dr=dr)))
        data_vecs += list(data_bvec)
    return np.array(data_vecs)

def eval_all_metrics(data_probs,data_str,bins=0):
    #data_probs = eval_prob_set(data_str)
    label_ids = eval(data_str).label_ids
    if len(label_ids)==0:
        return 0.,0.,0.,0.
    data_acc = fs.accuracy_score(np.argmax(data_probs,1),label_ids)
    data_loss = fs.cross_entropy(data_probs,label_ids)
    if bins==0:
        data_ece,data_oe = 0.,0.
    else:
        prob_per_bin,crrt_per_bin = calc_calibration(data_probs,label_ids,bins=bins)
        data_ece,data_oe = fs.calc_ece_and_oe(prob_per_bin,crrt_per_bin)
    return data_acc,data_loss,data_ece,data_oe

def eb_stop_metric(data_str, c_size=32):
    feed_size = len(eval(data_str).label_ids)
    fnum = int(np.ceil(feed_size/c_size))
    #fst_grads = sess.run(tf_grads, feed_dict=eval(str2exec_loss(0,c_size,data_str)))
    fst_grads = sess.run(tf_grads, feed_dict=eval(fs.str2exec(0,c_size,data_str)))
    
    all_grads = []
    for fst_grad in fst_grads:
        all_grads.append((c_size/feed_size)*fst_grad)
    
    for fi in range(1,fnum):
        common_str = data_str+".label_ids"+"["+str(fi)+"*"+str(c_size)
        common_str+= ":("+str(fi)+"+1)*"+str(c_size)+"]"
        fi_size = len(eval(common_str))
        #fi_grads = sess.run(tf_grads, feed_dict=eval(str2exec_loss(fi,c_size,data_str)))
        fi_grads = sess.run(tf_grads, feed_dict=eval(fs.str2exec(fi,c_size,data_str)))
        for ai in range(len(all_grads)):
            all_grads[ai] += (fi_size/feed_size)*fi_grads[ai]
    
    param_size,var_grads = 0,[]
    for all_grad in all_grads:
        var_grads.append(np.zeros_like(all_grad))
        one_size = 1
        for shp in np.shape(all_grad):
            one_size *= shp 
        param_size += one_size
    
    for di in range(feed_size):
        #smp_grads = sess.run(tf_grads, feed_dict=eval(str2exec_loss(di,1,data_str))) # tf_cpu_grads
        smp_grads = sess.run(tf_grads, feed_dict=eval(fs.str2exec(di,1,data_str))) # tf_cpu_grads
        for gi,(smp_grad,all_grad) in enumerate(zip(smp_grads,all_grads)):
            var_grads[gi] += (smp_grad-all_grad)**2
    
    for gi in range(len(var_grads)):
        var_grads[gi] /= (feed_size-1)
    
    sum_gradpw_de_var = 0.0
    for gi,(var_grad,all_grad) in enumerate(zip(var_grads,all_grads)):
        sum_gradpw_de_var += np.sum(all_grad**2 / (var_grad+1e-9))
    
    eb_crt = 1.0 - (sum_gradpw_de_var/param_size)*feed_size
    return eb_crt

data_path = './data/all/'+data
label_list = fs.data2label_list(data)
num_labels = len(label_list)

tf.reset_default_graph()
print (fs.get_available_gpus())

do_lower_case = True
curr_path = os.getcwd()
bert_dir = os.path.join(curr_path,os.pardir,'bert','uncased_L-12_H-768_A-12')
#bert_dir = os.path.join(curr_path,os.pardir,'bert','pretrain',data)

vocab_file = os.path.join(bert_dir, 'vocab.txt')
bert_config_file = os.path.join(bert_dir, 'bert_config.json')
init_checkpoint = os.path.join(bert_dir, 'bert_model.ckpt')
#init_checkpoint = os.path.join(bert_dir, 'model.ckpt-7500')
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

''' Modeling '''
''' # My setting '''
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
is_training = True
use_one_hot_embeddings = False

input_ids = tf.placeholder(tf.int32,[None,max_seq_length])
input_mask = tf.placeholder(tf.int32,[None,max_seq_length])
segment_ids = tf.placeholder(tf.int32,[None,max_seq_length]) 
labels = tf.placeholder(tf.int32,[None,]) 
learn_intf = tf.placeholder(tf.float32,[]) 
drop_rate = tf.placeholder(tf.float32, [], name='drop_rate')

# when testing, drop_rate = 0.0 
# when training, drop_rate = 0.1 
bert_config.hidden_dropout_prob = drop_rate # 0.1
bert_config.attention_probs_dropout_prob = drop_rate # 0.1

#with tf.device('/device:GPU:0'):
with tf.variable_scope('bert'):
    bert_enc = modeling.BertModel(
                  config=bert_config,
                  is_training=is_training,
                  input_ids=input_ids,
                  input_mask=input_mask,
                  token_type_ids=segment_ids,
                  use_one_hot_embeddings=use_one_hot_embeddings)

output_layer = bert_enc.get_pooled_output()
hidden_size = output_layer.shape[-1].value

output_weights = tf.get_variable("output_weights", 
                                 [num_labels, hidden_size],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
output_bias = tf.get_variable("output_bias", 
                              [num_labels], 
                              initializer=tf.zeros_initializer())

output_layer = tf.nn.dropout(output_layer, keep_prob=1.0-drop_rate)
logits = tf.matmul(output_layer, output_weights, transpose_b=True)
logits = tf.nn.bias_add(logits, output_bias)
log_probs = tf.nn.log_softmax(logits, axis=-1)
probs = tf.nn.softmax(logits, axis=-1)
preds = tf.argmax(log_probs,axis=1)
one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
#per_entropy_loss = -beta*tf.reduce_sum(probs*log_probs, axis=-1)
per_example_loss = -tf.reduce_sum(one_hot_labels*log_probs, axis=-1)
loss = tf.reduce_mean(per_example_loss)#+per_entropy_loss

vecs_local = tf.placeholder(tf.float32,[None,hidden_size])
logits_local = tf.matmul(vecs_local, output_weights, transpose_b=True)
logits_local = tf.nn.bias_add(logits_local, output_bias)
probs_local = tf.nn.softmax(logits_local, axis=-1)

''' For learning '''
tf_bert_opt = tf.train.AdamOptimizer(learning_rate=learn_intf)
tf_variables = tf.trainable_variables()
tf_bert_variables = tf_variables[1:]
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    tf_grads = tf.gradients(loss, tf_bert_variables)
    optimizer = tf_bert_opt.apply_gradients(zip(tf_grads,tf_bert_variables))

''' Training code '''
''' Load saved parameters '''
tvars = tf.trainable_variables()
(assignment_map, initialized_variable_names) = \
            modeling.get_assigment_map_from_checkpoint(tvars, init_checkpoint)
tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
print(len(assignment_map))

sess = tf.Session() #config=config)
sess.run(tf.global_variables_initializer())
emb_temp1 = sess.run(tvars[0])
sess.run(tf.global_variables_initializer())
emb_temp2 = sess.run(tvars[0])
print(emb_temp1)
print((emb_temp1==emb_temp2).all())
sess.close()

btchs = 16
dr = 0.2
lr = 0.00003
N_base = 1

pe_split = 4
pe_dev_ratio = 0.25
bert_save_path = './data/tmp_weights/bert_saved_gpu'+str(gpu)

saver = tf.train.Saver()
sub_path = data_path+'/logs/main_k'+str(k_shot)
if not os.path.isdir(sub_path):
    os.makedirs(sub_path)
si_str = '_si'+str(start)+'to'+str(end)
mylog = fs.my_logging(sub_path+'/main_k'+str(k_shot)+'_log_at_seed'+si_str+'.log')
init_weights = fs.load_all('./data/fin_layers/'+data+'_weights.bin')
# shape = (# of seed, N_base, # of class, dimension)

pre_dev_confs_over_seeds = []
pre_ulb_probs_over_seeds = []
pre_dev_probs_over_seeds = []
pre_dev_perfs_over_seeds = []
pre_pe_epochs_over_seeds = []

all_records_over_seeds = []
all_u_probs_over_seeds = []

for si in range(start,end):
    full_trn_examples = fs.tsv2examples(os.path.join(data_path,'train.tsv'))
    tst_examples = fs.tsv2examples(os.path.join(data_path,'test.tsv'))
    
    rand_indx = [i for i in range(len(full_trn_examples))]
    random.Random(si).shuffle(rand_indx)
    
    mylog.printNlogging(' ')
    mylog.printNlogging(data)
    mylog.printNlogging('si:',si)
    mylog.printNlogging('Random_indx:',np.array(rand_indx))
    mylog.printNlogging(' ')
    
    examples_per_class = {label:[] for label in label_list}
    vlp_examples_per_class = {label:[] for label in label_list}
    for ri in rand_indx:
        example = full_trn_examples[ri]
        label = example.label
        if k_shot > len(examples_per_class[label]):
            examples_per_class[label].append(example)
        elif valid_k > len(vlp_examples_per_class[label]):
            vlp_examples_per_class[label].append(example)
    
    trn_examples,vlp_examples = [],[]
    for label in label_list:
        trn_examples += examples_per_class[label]
        vlp_examples += vlp_examples_per_class[label]
    
    trn_features = fs.convert_examples_to_features(trn_examples, label_list, max_seq_length, tokenizer)
    tst_features = fs.convert_examples_to_features(tst_examples, label_list, max_seq_length, tokenizer)
    vlp_features = fs.convert_examples_to_features(vlp_examples, label_list, max_seq_length, tokenizer)
    
    trn_org = fs.buildTensorInputs(trn_features)
    tst = fs.buildTensorInputs(tst_features)
    ulb = fs.buildTensorInputs(tst_features)
    vlp = fs.buildTensorInputs(vlp_features)
    
    'Pre-running stage'
    'calculate pe-stop-epoch & confidence distribution & val-n stop performance'
    pre_ulb_probs = []
    pre_dev_probs = []
    pre_dev_perfs = []
    pe_stop_epoch = []
    samples_conf_mat = np.zeros((pe_split,len(trn_examples)))
    
    examples_per_class = {label:[] for label in label_list}
    for example in trn_examples:
        examples_per_class[example.label].append(example)
    evenly_examples = fs.evenly_shuffle([examples_per_class[label] for label in label_list])
    evenly_features = fs.convert_examples_to_features(evenly_examples, label_list, max_seq_length, tokenizer)
    
    lbd_len = len(evenly_features)
    pe_trn_len = int(lbd_len*(1.-pe_dev_ratio) + 0.5)
    p_step = lbd_len//pe_split
    for pi in range(pe_split):
        trn_indices = [i%lbd_len for i in range(pi*p_step,pi*p_step+pe_trn_len)]
        dev_indices = list(set(np.arange(lbd_len))-set(trn_indices))
        
        pe_trn_features = fs.select_by_index(evenly_features,trn_indices)
        pe_dev_features = fs.select_by_index(evenly_features,dev_indices)
        
        pe_trn = fs.buildTensorInputs(pe_trn_features)
        pe_dev = fs.buildTensorInputs(pe_dev_features)
        
        pe_trn_len,pe_dev_len = len(pe_trn_features),len(pe_dev_features)
        
        start_t = time.time()
        pi_stop_epoch = 0
        dev_best = np.inf
        for base_i in range(N_base):
            sess = tf.Session() #
            sess.run(tf.global_variables_initializer())
            sess.run(output_weights.assign(init_weights[si][base_i]))
            
            trn = fs.build_trn_set(pe_trn,epochs)
            trn_bnum = int(np.ceil(len(trn.label_ids)/btchs))
            
            mylog.printNlogging('PE-calculating...')
            mylog.printNlogging(data)
            mylog.printNlogging('(trn_len,dev_len,tst_len,ulb_len) =',
                                (pe_trn_len,pe_dev_len,len(tst_examples),len(tst_examples)))
            mylog.printNlogging('seed:',si)
            mylog.printNlogging('pi:',pi,'/',pe_split)
            mylog.printNlogging('base_i :',base_i,'/',N_base)
            mylog.printNlogging('wght_1 :',init_weights[si][base_i][0][0])
            
            epoch = 0
            current_best, patience = np.inf, 0
            for bi in range(trn_bnum):
                if (bi*btchs)//pe_trn_len==epoch:
                    trn_acc,trn_loss = eval_acc_loss('pe_trn')
                    dev_acc,dev_loss = eval_acc_loss('pe_dev')
                    tst_acc,tst_loss = 0.,0.#eval_acc_loss('tst')
                    
                    if dev_loss < current_best:
                        current_best = dev_loss
                        patience = 0
                    else:
                        patience += 1
                        if patience > 8:
                            break
                    if dev_loss < dev_best:
                        dev_best = dev_loss
                        pi_stop_epoch = epoch
                        dev_confs = eval_prob_set('pe_dev',dr=0.).max(1)
                        saver.save(sess, bert_save_path)
                                
                    mylog.printNlogging(epoch, 'epoch,',
                          round((time.time() - start_t)/60.0, 4),'m //',
                          '[trn_l,dev_l,tst_l] =', list(np.around([trn_loss,dev_loss,tst_loss],4)),'//',
                          '[trn_a,dev_a,tst_a] =', list(np.around([trn_acc ,dev_acc ,tst_acc ],4)),)
                    epoch += 1
                    prv_dev_loss = dev_loss
                    
                if epoch > epochs: 
                    break
                _, cost = sess.run([optimizer, loss], feed_dict=eval(fs.str2exec(bi,btchs,'trn',lr,dr)))
            
            sess.close() 
            mylog.printNlogging(50*'#')
            mylog.printNlogging(' ')
            
            if pi==0 and base_i==0:
                sess = tf.Session() #
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, bert_save_path)
                tst_probs = eval_prob_set('tst')
                l_tst_acc,l_tst_loss,l_tst_ece,l_tst_oe = eval_all_metrics(tst_probs,'tst',bins=15)
                print(' ')
                print('val-stop local tst acc/loss/ece/oe =',
                      fs.rounds([l_tst_acc,l_tst_loss,l_tst_ece,l_tst_oe]))
                print(' ')
                sess.close()
        
        sess = tf.Session() #
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, bert_save_path)
        ulb_probs = eval_prob_set('ulb')
        pre_ulb_probs.append(ulb_probs)
        pre_dev_probs.append(eval_prob_set('pe_dev'))
        pre_dev_perfs.append(list(eval_acc_loss('pe_dev')))
        if pi==0:
            tst_probs = ulb_probs
            g_tst_acc,g_tst_loss,g_tst_ece,g_tst_oe = eval_all_metrics(tst_probs,'tst',bins=15)
            #g_tst_acc,g_tst_loss = eval_acc_loss('tst')
            print(' ')
            print('val-stop global tst acc/loss/ece/oe =',
                  fs.rounds([g_tst_acc,g_tst_loss,g_tst_ece,g_tst_oe]))
            print(' ')
        sess.close()
        
        pe_stop_epoch.append(pi_stop_epoch)
        for dev_i,dev_conf in zip(dev_indices,dev_confs):
            samples_conf_mat[pi,dev_i] = dev_conf
    
    'seed i results'
    pe_stop_epoch = np.mean(pe_stop_epoch)
    samples_confs = samples_conf_mat.sum(0) / (samples_conf_mat!=0.0).astype('float32').sum(0)
    
    pre_dev_confs_over_seeds.append(samples_confs)
    pre_ulb_probs_over_seeds.append(pre_ulb_probs)
    pre_dev_probs_over_seeds.append(pre_dev_probs)
    pre_dev_perfs_over_seeds.append(pre_dev_perfs)
    pre_pe_epochs_over_seeds.append(int(pe_stop_epoch+0.5))
    
    fs.dump_all(np.array(pre_dev_confs_over_seeds),sub_path+'/main_pre_dev_confs'+si_str+'.bin')
    fs.dump_all(np.array(pre_ulb_probs_over_seeds),sub_path+'/main_pre_ulb_probs'+si_str+'.bin')
    fs.dump_all(np.array(pre_dev_probs_over_seeds),sub_path+'/main_pre_dev_probs'+si_str+'.bin')
    fs.dump_all(np.array(pre_dev_perfs_over_seeds),sub_path+'/main_pre_dev_perfs'+si_str+'.bin')
    fs.dump_all(np.array(pre_pe_epochs_over_seeds),sub_path+'/main_pre_pe_epochs'+si_str+'.bin')
    
    samples_confs = np.sort(samples_confs)
    mylog.printNlogging('pe_stop_epoch :',int(pe_stop_epoch+0.5))
    val_per_class = str(int(len(trn_examples)*pe_dev_ratio/num_labels))
    print('val-'+val_per_class+'-stop local tst acc/loss/ece/oe =',fs.rounds([l_tst_acc,l_tst_loss,l_tst_ece,l_tst_oe]))
    print('val-'+val_per_class+'-stop global tst acc/loss/ece/oe =',fs.rounds([g_tst_acc,g_tst_loss,g_tst_ece,g_tst_oe]))
    print(50*'#')
    print(' ')
    
    'All labeled samples training and stop-method comparison:'
    'EB, PE, Val+25, BUS (ent,conf), LID'
    trn_len,ulb_len,tst_len,vlp_len = \
        len(trn_features),len(tst_features),len(tst_features),len(vlp_features)
    
    if ulb_len > 10000: # to reduce the time complexity of the LID
        ulb_snip_ids = np.random.choice(ulb_len,10000,False)
    else:
        ulb_snip_ids = np.arange(ulb_len)
    
    start_t = time.time()    
    dev_best = np.inf
    all_records_over_bases = []
    all_u_probs_over_bases = []
    for base_i in range(N_base):
        sess = tf.Session() #
        sess.run(tf.global_variables_initializer())
        sess.run(output_weights.assign(init_weights[si][base_i]))
        
        trn = fs.build_trn_set(trn_org,epochs)
        trn_bnum = int(np.ceil(len(trn.label_ids)/btchs))
        
        mylog.printNlogging('Stop-method Comparison...')
        mylog.printNlogging(data)
        mylog.printNlogging('(trn_len,dev_len,ulb_len,tst_len) =',(trn_len,vlp_len,ulb_len,tst_len))
        mylog.printNlogging('seed:',si)
        mylog.printNlogging('base_i :',base_i,'/',N_base)
        mylog.printNlogging('wght_1 :',init_weights[si][base_i][0][0])
        
        print_columns = ['epoch','time(m)','trn_a','trn_l',
                         'tst_a','tst_l','tst_ece','tst_oe',
                         'es_dva','es_dvl','es_cnf','es_eb',
                         'es_lm5','es_lm10','es_lm20','es_lm50','es_lm1h']
        row_format ="{:<9}" * (len(print_columns))
        mylog.printNlogging(row_format.format(*print_columns))
        
        epoch = 0
        eb_rcd = deque(3*[0], 3)
        records = []
        u_probs_over_epochs = []
        for bi in range(trn_bnum):
            if (bi*btchs)//trn_len==epoch:
                trn_acc,trn_loss = eval_acc_loss('trn_org')
                dev_acc,dev_loss = eval_acc_loss('vlp')
                
                ulb_vecs = eval_vec_set('ulb',dr=0.0)
                ulb_probs = sess.run(probs_local, feed_dict={vecs_local:ulb_vecs})
                ulb_confs = np.sort(ulb_probs.max(1))
                #ulb_probs = eval_prob_set('ulb',dr=0.)
                
                tst_probs = ulb_probs#eval_prob_set('tst')
                tst_acc,tst_loss,tst_ece,tst_oe = eval_all_metrics(tst_probs,'tst',bins=15)
                
                # prefex 'es_' indicates a criterion for ealry stopping 
                'LID stopping: the lowest is the stop-point'
                eud_mat = pairwise_distances(ulb_vecs[ulb_snip_ids],metric='euclidean')
                es_lids = [fs.LID(eud_mat,m=m) for m in [5,10,20,50,100]]
                
                'Confidence stopping: the lowest is the stop-point'
                ids = np.arange(0,ulb_len,ulb_len/len(samples_confs)).astype('int32')
                es_cnf = np.linalg.norm(ulb_confs[ids]-samples_confs)
                
                'Dev-based stopping'
                es_dvl,es_dva = dev_loss,dev_acc
                
                'EB stopping: first over zero value is the stop-point'
                if sum(eb_rcd)<len(eb_rcd):
                    trn_for_eb = fs.select_by_index(trn_features,np.random.choice(trn_len,64,False))
                    trn_for_eb = fs.buildTensorInputs(trn_for_eb)
                    es_eb = eb_stop_metric('trn_for_eb', c_size=btchs)
                else:
                    es_eb = 1.
                
                if es_eb>0: 
                    eb_rcd.append(1)
                else:
                    eb_rcd.append(0)
                
                if dev_loss < dev_best:
                    dev_best = dev_loss
                    vg_tst_acc,vg_tst_loss,vg_tst_ece,vg_tst_oe = tst_acc,tst_loss,tst_ece,tst_oe
                    if base_i==0:
                        vl_tst_acc,vl_tst_loss,vl_tst_ece,vl_tst_oe = tst_acc,tst_loss,tst_ece,tst_oe
                
                record = [epoch,round((time.time() - start_t)/60.0,2),
                          trn_acc,trn_loss,tst_acc,tst_loss,tst_ece,tst_oe,
                          es_dva,es_dvl,es_cnf,es_eb]+es_lids
                rounded_record = fs.rounds(record)
                mylog.printNlogging(row_format.format(*rounded_record))
                records.append(record)
                u_probs_over_epochs.append(ulb_probs)
                
                epoch += 1
                
            if epoch > epochs: 
                break
            _, cost = sess.run([optimizer, loss], feed_dict=eval(fs.str2exec(bi,btchs,'trn',lr,dr)))
        
        sess.close() 
        all_records_over_bases.append(records)
        all_u_probs_over_bases.append(u_probs_over_epochs)
        mylog.printNlogging(50*'#')
        mylog.printNlogging(' ')
    all_records_over_seeds.append(all_records_over_bases)
    all_u_probs_over_seeds.append(all_u_probs_over_bases)
    
    fs.dump_all(np.array(all_records_over_seeds),sub_path+'/main_all_records'+si_str+'.bin')
    fs.dump_all(np.array(all_u_probs_over_seeds),sub_path+'/main_ulb_records'+si_str+'.bin')
    
    mylog.printNlogging(' ')
    mylog.printNlogging(50*'#')
    mylog.printNlogging('Result_for_seed =',si)
    mylog.printNlogging('pe_stop_epoch =',int(pe_stop_epoch+0.5))
    val_per_class = str(int(len(trn_examples)*pe_dev_ratio/num_labels))
    mylog.printNlogging('val-'+val_per_class+'-stop local tst acc/loss/ece/oe =',fs.rounds([l_tst_acc,l_tst_loss,l_tst_ece,l_tst_oe]))
    mylog.printNlogging('val-'+val_per_class+'-stop global tst acc/loss/ece/oe =',fs.rounds([g_tst_acc,g_tst_loss,g_tst_ece,g_tst_oe]))
    
    val_per_class = str(int(vlp_len/num_labels))
    mylog.printNlogging('val+'+val_per_class+'-stop local tst acc/loss/ece/oe =',fs.rounds([vl_tst_acc,vl_tst_loss,vl_tst_ece,vl_tst_oe]))
    mylog.printNlogging('val+'+val_per_class+'-stop global tst acc/loss/ece/oe =',fs.rounds([vg_tst_acc,vg_tst_loss,vg_tst_ece,vg_tst_oe]))
    mylog.printNlogging(50*'#')
    mylog.printNlogging(' ')
