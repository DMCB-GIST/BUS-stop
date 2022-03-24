# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 03:25:43 2021

@author: Anonymous

functions
"""

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle

print(os.getcwd())

def rounds(lst,cut=4):
    return list(np.around(np.array(lst),cut))

def dump_all(all_data, data_path):
    with open(data_path, 'wb') as handle:
        pickle.dump(all_data, handle)

def load_all(data_path):
    with open(data_path, 'rb') as handle:
        load_all = pickle.loads(handle.read())
    return load_all

def find_eb_stop(ebs):
    for i,eb in enumerate(ebs):
        if eb==1.:
            break
    return max(i-3,1)

def after_max_of_min_k(maxs,mins,k=5):
    maxs,mins = np.array(maxs),np.array(mins)
    min_i = np.argmin(mins)
    min_ks = np.arange(min_i,min(min_i+k,len(mins)))
    return min_ks[np.argmax(maxs[min_ks])]

def class_calibration(observed_dist,sampling_bias,val_acc):
    sampling_bias = np.array(sampling_bias)/np.sum(sampling_bias)
    observed_dist = np.array(observed_dist)/np.sum(observed_dist)
    acc_min = 1./len(sampling_bias)
    frac = (1.-acc_min) / (val_acc-acc_min)
    calibrated_dist = sampling_bias + frac*(observed_dist-sampling_bias)
    if (calibrated_dist>1.0).any(): # clipped for easy analysis, but clipping does not affect to the results.
        new_calibrated_dist = np.zeros_like(calibrated_dist)
        new_calibrated_dist[calibrated_dist.argmax()] = 1.0
        calibrated_dist = new_calibrated_dist
    return calibrated_dist

your_path = '.'

### Table 2 Results ###
print(' ')
print(50*'#')

data = 'SST-2' # ['SST-2','IMDB','Elec','AGNews','DBPedia']
K = 50 # [50,100,200,400,800,1600]
data_path = your_path+'/data/all/'+data

f = open(data_path+'/logs/main_k'+str(K)+'/main_k'+str(K)+'_log_at_seed_si0to10.log','r')
val_splits = []
for line in f.readlines():
    if 'val-' in line and '-stop local' in line:
        val_splits.append(eval(line.split('=')[-1].strip()))
f.close()

pre_dev_confs_over_seeds = load_all(data_path+'/logs/main_k'+str(K)+'/main_pre_dev_confs_si0to10.bin')
pre_ulb_probs_over_seeds = load_all(data_path+'/logs/main_k'+str(K)+'/main_pre_ulb_probs_si0to10.bin')
pre_dev_probs_over_seeds = load_all(data_path+'/logs/main_k'+str(K)+'/main_pre_dev_probs_si0to10.bin')
pre_dev_perfs_over_seeds = load_all(data_path+'/logs/main_k'+str(K)+'/main_pre_dev_perfs_si0to10.bin')
pre_pe_epochs_over_seeds = load_all(data_path+'/logs/main_k'+str(K)+'/main_pre_pe_epochs_si0to10.bin')
all_records_over_seeds = load_all(data_path+'/logs/main_k'+str(K)+'/main_all_records_si0to10.bin')
all_u_probs_over_seeds = load_all(data_path+'/logs/main_k'+str(K)+'/main_ulb_records_si0to10.bin')

seeds,N_base,epochs,_ = np.shape(all_records_over_seeds)
perfs_stop_method = defaultdict(list)
for si in range(seeds):

    sample_confs = sorted(pre_dev_confs_over_seeds[si])
    pe_stop_epoch = pre_pe_epochs_over_seeds[si]
    T = len(pre_ulb_probs_over_seeds[si])
    
    pre_ulb_dists,cali_dists = [],[]
    for ti in range(T):
        val_acc,val_loss = pre_dev_perfs_over_seeds[si][ti]
        val_dist = pre_dev_probs_over_seeds[si][ti].mean(0)

        pre_ulb_dist = pre_ulb_probs_over_seeds[si][ti].mean(0)
        cali_dist = class_calibration(pre_ulb_dist,val_dist,val_acc) # F1 == Acc

        pre_ulb_dists.append(pre_ulb_dist)
        cali_dists.append(cali_dist)

    pre_ulb_dist = np.mean(pre_ulb_dists,0) # hat{C_u}
    cali_dist = np.mean(cali_dists,0) # vec{C_u}

    for base_i in range(N_base):

        all_ulb_dist = all_u_probs_over_seeds[si][base_i].mean(1)
        tst_perfs = all_records_over_seeds[si][base_i][:,4:8]
        dev_accs = all_records_over_seeds[si][base_i][:,8]
        dev_losses = all_records_over_seeds[si][base_i][:,9]
        conf_sims = all_records_over_seeds[si][base_i][:,10]
        eb_vals = all_records_over_seeds[si][base_i][:,11]
        lid_m05s = all_records_over_seeds[si][base_i][:,12]
        lid_m10s = all_records_over_seeds[si][base_i][:,13]
        lid_m20s = all_records_over_seeds[si][base_i][:,14]
        lid_m50s = all_records_over_seeds[si][base_i][:,15]
        lid_m1hs = all_records_over_seeds[si][base_i][:,16]
        class_sims = cosine_similarity([cali_dist],all_ulb_dist)[0]

        dev_acc_i = np.argmax(dev_accs)
        dev_loss_i = np.argmin(dev_losses)
        conf_sim_i = np.argmin(conf_sims)
        eb_val_i = find_eb_stop(eb_vals)
        lid_m05_i = np.argmin(lid_m05s)
        lid_m10_i = np.argmin(lid_m10s)
        lid_m20_i = np.argmin(lid_m20s)
        lid_m50_i = np.argmin(lid_m50s)
        lid_m1h_i = np.argmin(lid_m1hs)
        class_sim_i = np.argmax(class_sims)
        bus_i = after_max_of_min_k(class_sims,conf_sims,k=5)
        
        perfs_stop_method['EB       '].append(tst_perfs[eb_val_i])
        perfs_stop_method['LID      '].append(tst_perfs[lid_m1h_i])
        perfs_stop_method['PE       '].append(tst_perfs[pe_stop_epoch])
        perfs_stop_method['Conf_sim '].append(tst_perfs[conf_sim_i])
        perfs_stop_method['Class_sim'].append(tst_perfs[class_sim_i])
        perfs_stop_method['BUS_stop '].append(tst_perfs[bus_i])
        perfs_stop_method['Val_add  '].append(tst_perfs[dev_loss_i])

print(' ')
print(data)
print('Balanced Classification')
print('K =', K)
print(' ')
print_columns = ['Method','Acc','Loss','ECE','OE']
row_format ="{:<12}" * (len(print_columns))
print(row_format.format(*print_columns))
row = ['Val_split']+rounds(np.array(val_splits).mean(0))
print(row_format.format(*row))
for method in perfs_stop_method:
    row = [method]+rounds(rounds(np.array(perfs_stop_method[method]).mean(0)))
    print(row_format.format(*row))


### Table 4 Results ###
print(' ')
print(' ')
print(50*'#')
'Imbalanced Setting'
data = 'SST-2' # ['SST-2','IMDB','Elec']
trn_neg = 0.5
tst_neg = 0.2
    
base_path = your_path+'/data/all/'+data+'/logs/'+'K100-TR'+str(trn_neg)+'-TS'+str(tst_neg)
pre_dev_confs_over_seeds = load_all(base_path+'/skewed_pre_dev_confs_si0to10.bin')
pre_ulb_probs_over_seeds = load_all(base_path+'/skewed_pre_ulb_probs_si0to10.bin')
pre_dev_probs_over_seeds = load_all(base_path+'/skewed_pre_dev_probs_si0to10.bin')
pre_dev_perfs_over_seeds = load_all(base_path+'/skewed_pre_dev_perfs_si0to10.bin')
pre_pe_epochs_over_seeds = load_all(base_path+'/skewed_pre_pe_epochs_si0to10.bin')
all_records_over_seeds = load_all(base_path+'/skewed_all_records_si0to10.bin')
all_u_probs_over_seeds = load_all(base_path+'/skewed_ulb_records_si0to10.bin')

f = open(base_path+'/skewed_log_at_seed_si0to10.log','r')
lines = f.readlines()
val_splits = []
for line in lines:
    if 'val-25-stop local' in line:
        val_splits.append(eval(line.split('=')[-1].strip()))
f.close()
        
seeds,N_base,epochs,_ = np.shape(all_records_over_seeds)
perfs_stop_method = defaultdict(list)
for si in range(seeds):

    sample_confs = sorted(pre_dev_confs_over_seeds[si])
    pe_stop_epoch = pre_pe_epochs_over_seeds[si]

    T = len(pre_ulb_probs_over_seeds[si])
    pre_ulb_dists,cali_dists = [],[]
    for ti in range(T):
        val_acc,val_f1,val_loss = pre_dev_perfs_over_seeds[si][ti]
        val_dist = pre_dev_probs_over_seeds[si][ti].mean(0)

        pre_ulb_dist = pre_ulb_probs_over_seeds[si][ti].mean(0)
        cali_dist = class_calibration(pre_ulb_dist,val_dist,val_f1)

        pre_ulb_dists.append(pre_ulb_dist)
        cali_dists.append(cali_dist)

    pre_ulb_dist = np.mean(pre_ulb_dists,0)
    cali_dist = np.mean(cali_dists,0)

    perfs_local_stop = defaultdict(list)
    perfs_global_criterion = defaultdict(list)
    for base_i in range(N_base):
        all_ulb_dist = all_u_probs_over_seeds[si][base_i].mean(1)
        tst_perfs = all_records_over_seeds[si][base_i][:,4:9]
        dev_accs = all_records_over_seeds[si][base_i][:,9]
        dev_losses = all_records_over_seeds[si][base_i][:,10]
        conf_sims = all_records_over_seeds[si][base_i][:,11]
        eb_vals = all_records_over_seeds[si][base_i][:,12]
        lid_m05s = all_records_over_seeds[si][base_i][:,13]
        lid_m10s = all_records_over_seeds[si][base_i][:,14]
        lid_m20s = all_records_over_seeds[si][base_i][:,15]
        lid_m50s = all_records_over_seeds[si][base_i][:,16]
        lid_m1hs = all_records_over_seeds[si][base_i][:,17]
        class_sims = cosine_similarity([cali_dist],all_ulb_dist)[0]

        dev_acc_i = np.argmax(dev_accs)
        dev_loss_i = np.argmin(dev_losses)
        conf_sim_i = np.argmin(conf_sims)
        eb_val_i = find_eb_stop(eb_vals)
        lid_m05_i = np.argmin(lid_m05s)
        lid_m10_i = np.argmin(lid_m10s)
        lid_m20_i = np.argmin(lid_m20s)
        lid_m50_i = np.argmin(lid_m50s)
        lid_m1h_i = np.argmin(lid_m1hs)
        class_sim_i = np.argmax(class_sims)
        bus_i = after_max_of_min_k(class_sims,conf_sims,k=5)

        perfs_stop_method['EB       '].append(tst_perfs[eb_val_i])
        perfs_stop_method['LID      '].append(tst_perfs[lid_m1h_i])
        perfs_stop_method['PE       '].append(tst_perfs[pe_stop_epoch])
        perfs_stop_method['Conf_sim '].append(tst_perfs[conf_sim_i])
        perfs_stop_method['Class_sim'].append(tst_perfs[class_sim_i])
        perfs_stop_method['BUS_stop '].append(tst_perfs[bus_i])
        perfs_stop_method['Val_add  '].append(tst_perfs[dev_loss_i])

print(' ')
print(data)
print('Imbalanced Classification')
print('n_l = 100')
print(' ')
print_columns = ['Method','Acc', 'F1', 'Loss','ECE','OE']
row_format ="{:<12}" * (len(print_columns))
print(row_format.format(*print_columns))
row = ['Val_split']+rounds(np.array(val_splits).mean(0))
print(row_format.format(*row))
for method in perfs_stop_method:
    row = [method]+rounds(rounds(np.array(perfs_stop_method[method]).mean(0)))
    print(row_format.format(*row))

'''
### Table 5 Results ###
print(' ')
print(' ')
print(50*'#')

data = 'SST-2'
K = 100

skewed_d = defaultdict(list)
for trn_neg,tst_neg in [[0.2,0.2],[0.2,0.4],[0.2,0.6],[0.2,0.8], [0.4,0.2],[0.4,0.4],[0.4,0.6],[0.4,0.8], 
                         [0.6,0.2],[0.6,0.4],[0.6,0.6],[0.6,0.8], [0.8,0.2],[0.8,0.4],[0.8,0.6],[0.8,0.8],]:
    base_path = your_path+'/data/all/'+data+'/logs/K100-TR'+str(trn_neg)+'-TS'+str(tst_neg)

    pre_dev_confs_over_seeds = load_all(base_path+'/skewed_pre_dev_confs_si0to10.bin')
    pre_ulb_probs_over_seeds = load_all(base_path+'/skewed_pre_ulb_probs_si0to10.bin')
    pre_dev_probs_over_seeds = load_all(base_path+'/skewed_pre_dev_probs_si0to10.bin')
    pre_dev_perfs_over_seeds = load_all(base_path+'/skewed_pre_dev_perfs_si0to10.bin')
    pre_pe_epochs_over_seeds = load_all(base_path+'/skewed_pre_pe_epochs_si0to10.bin')
    all_records_over_seeds = load_all(base_path+'/skewed_all_records_si0to10.bin')
    all_u_probs_over_seeds = load_all(base_path+'/skewed_ulb_records_si0to10.bin')
    seeds,N_base,epochs,_ = np.shape(all_records_over_seeds)
    
    skewed_dd = defaultdict(list)
    si_pred_dists,si_cali_f1s,si_cali_accs = [],[],[]
    for si in range(seeds):
        sample_confs = sorted(pre_dev_confs_over_seeds[si])
        pe_stop_epoch = pre_pe_epochs_over_seeds[si]

        T = len(pre_ulb_probs_over_seeds[si])
        pre_ulb_dists,cali_dist_f1s,cali_dist_accs = [],[],[]
        for ti in range(T):
            val_acc,val_f1,val_loss = pre_dev_perfs_over_seeds[si][ti]
            val_dist = pre_dev_probs_over_seeds[si][ti].mean(0)

            pre_ulb_dist = pre_ulb_probs_over_seeds[si][ti].mean(0)
            cali_dist_f1 = class_calibration(pre_ulb_dist,val_dist,val_f1)
            cali_dist_acc = class_calibration(pre_ulb_dist,val_dist,val_acc)

            pre_ulb_dists.append(pre_ulb_dist)
            cali_dist_f1s.append(cali_dist_f1)
            cali_dist_accs.append(cali_dist_acc)

        pre_ulb_dist = np.mean(pre_ulb_dists,0)
        cali_dist_f1 = np.mean(cali_dist_f1s,0)
        cali_dist_acc = np.mean(cali_dist_accs,0)

        si_pred_dists.append(pre_ulb_dist)
        si_cali_f1s.append(cali_dist_f1)
        si_cali_accs.append(cali_dist_acc)

        for base_i in range(N_base):

            all_ulb_dist = all_u_probs_over_seeds[si][base_i].mean(1)
            tst_perfs = all_records_over_seeds[si][base_i][:,4:9]
            dev_accs = all_records_over_seeds[si][base_i][:,9]
            dev_losses = all_records_over_seeds[si][base_i][:,10]
            conf_sims = all_records_over_seeds[si][base_i][:,11]
            eb_vals = all_records_over_seeds[si][base_i][:,12]
            lid_m05s = all_records_over_seeds[si][base_i][:,13]
            lid_m10s = all_records_over_seeds[si][base_i][:,14]
            lid_m20s = all_records_over_seeds[si][base_i][:,15]
            lid_m50s = all_records_over_seeds[si][base_i][:,16]
            lid_m1hs = all_records_over_seeds[si][base_i][:,17]
            class_sims_p = cosine_similarity([pre_ulb_dist],all_ulb_dist)[0]
            class_sims_f = cosine_similarity([cali_dist_f1],all_ulb_dist)[0]
            class_sims_a = cosine_similarity([cali_dist_acc],all_ulb_dist)[0]

            dev_acc_i = np.argmax(dev_accs)
            dev_loss_i = np.argmin(dev_losses)
            conf_sim_i = np.argmin(conf_sims)
            eb_val_i = find_eb_stop(eb_vals)
            lid_m05_i = np.argmin(lid_m05s)
            lid_m10_i = np.argmin(lid_m10s)
            lid_m20_i = np.argmin(lid_m20s)
            lid_m50_i = np.argmin(lid_m50s)
            lid_m1h_i = np.argmin(lid_m1hs)
            class_simp_i = np.argmax(class_sims_p)
            class_simf_i = np.argmax(class_sims_f)
            class_sima_i = np.argmax(class_sims_a)
            #bus_i = after_max_of_min_k(class_sims_p,conf_sims,k=5) # non-cali pred, hat{C_u}
            #bus_i = after_max_of_min_k(class_sims_a,conf_sims,k=5) # cali, B = Acc_val
            bus_i = after_max_of_min_k(class_sims_f,conf_sims,k=5) # cali, B = Acc_f1

            skewed_dd['EB'].append(tst_perfs[eb_val_i])
            skewed_dd['Val_add'].append(tst_perfs[dev_loss_i])
            skewed_dd['BUS_stop'].append(tst_perfs[bus_i])

    metric_id = 0 # 0 for accuracy, 1 for F1, 2 for Loss
    for method in skewed_dd:
        skewed_d[method].append(np.array(skewed_dd[method]).mean(0)[metric_id])

print(' ')
print(data)
print('Various imbalanced settings')
for m in skewed_d:
    print(' ')
    for row in np.reshape(skewed_d[m],(4,4)):
        print(' '.join(map(str, np.around(row,3))))
    print(m, '= AVG:',round(np.mean(skewed_d[m]),4))
print(' ')
'''
print(50*'#')
print(' ')

