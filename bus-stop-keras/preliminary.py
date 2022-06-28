"""
First author: HongSeok Choi (hongking9@gist.ac.kr)
Corresponding author: Hyunju Lee (hyunjulee@gist.ac.kr)
Code for Early Stopping Based on Unlabeled Samples in Text Classification.
"""
import numpy as np
import logging
import tensorflow as tf
from preprocessing import evenly_shuffle, select_by_index

from tensorflow.keras.activations import softmax
from sklearn.utils import shuffle
from sklearn.metrics import log_loss, f1_score, accuracy_score

logger = logging.getLogger('BUS-stop')

def class_calibration(unl_dist,smp_bias,val_perf):
    '''val_perf is F1-score or Accuracy'''
    assert(len(unl_dist)==len(smp_bias))
    num_labels = len(smp_bias)
    smp_bias = np.array(smp_bias)/np.sum(smp_bias)
    unl_dist = np.array(unl_dist)/np.sum(unl_dist)
    
    acc_min = 1./num_labels
    frac = (1.-acc_min) / (val_perf-acc_min)
    cali_dist = smp_bias + frac*(unl_dist-smp_bias)
    cali_dist = np.clip(cali_dist, 0., 1.)
    return cali_dist

def run_stage(strategy, modeler, processor, lab_examples, X_unl, rand_seed=0,
			  epochs=30, patience=10, batch_size=16, learning_rate=3e-5, 
			  val_ratio=0.5, T=5, n_base=5, verbose=1):
			  #, cali_acc_or_f1='f1', bias_lab_or_val='val'):
	gpus = strategy.num_replicas_in_sync
	y_lab = np.array([processor.label_map[ex.label] for ex in lab_examples])
	label_list = processor.get_label_list()
	lab_len = len(lab_examples)
	num_labels = len(label_list)

	examples_per_class = {label:[] for label in label_list}
	for ex in shuffle(lab_examples, random_state=rand_seed):
		examples_per_class[ex.label].append(ex)
	evenly_lab_examples = evenly_shuffle([examples_per_class[label] for label in label_list])
	logger.debug('Labels in the labeled set mixed evenly like this, {}.'.format(
		[ex.label for ex in evenly_lab_examples[:2*len(label_list)]]+['...']))

	trn_len = int(round(lab_len*(1.-val_ratio),0))
	steps_per_epoch = trn_len//batch_size # We drop the last batch, the size of which is smaller.
	t_unit = lab_len//T

	sample_conf_mat = np.zeros((T,lab_len)) 
	''' Record all the prediction results obtained in preliminary stage '''
	preliminary_records = {'val_acc':[], 
					    'val_f1':[],
						'val_dist':[], # class distribution predicted on the validation set
						'ulb_dist':[], # class distribution predicted on the unlabeled set
						'lab_dist':[], # class distribution of the labeled set
						'sample_confs':None} # the prediction confidences for each sample in labeled data
	preliminary_records['lab_dist'] = np.mean(np.eye(np.max(y_lab)+1)[y_lab],0)
	for t in range(T):
		logger.debug(' ')
		logger.debug('{}-th run / total {} runs'.format(t,T))
		
		trn_indices = [i%lab_len for i in range(t*t_unit,t*t_unit+trn_len)]
		val_indices = list(set(np.arange(lab_len))-set(trn_indices))
		
		trn_examples = select_by_index(evenly_lab_examples,trn_indices)
		val_examples = select_by_index(evenly_lab_examples,val_indices)
		
		X_trn,y_trn = processor.examples_to_features(trn_examples)
		X_val,y_val = processor.examples_to_features(val_examples)
		
		best_val_loss, best_val_acc = np.inf, 0.0
		for base_i in range(n_base):
			with strategy.scope():
				model = modeler.build_model()
				model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08), 
							  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
							  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])

			model.fit(x=X_trn, y=y_trn, shuffle=True, epochs=epochs, steps_per_epoch=steps_per_epoch, 
					  validation_data=(X_val, y_val), batch_size=batch_size*gpus, verbose=verbose,
					  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)])

			val_loss, val_acc = model.evaluate(X_val, y_val, verbose=verbose)
			logger.debug("For base_i {}, [val_acc,val_loss] = [{},{}].".format(base_i,round(val_acc,4),round(val_loss,4)))

			if val_loss < best_val_loss: # val_acc > best_val_acc
				best_weights = model.get_weights() 
				best_val_loss = val_loss
				
		model.set_weights(best_weights)
		
		val_probs = softmax(model(X_val)).numpy()
		val_acc = accuracy_score(y_val, val_probs.argmax(1))
		val_f1 = f1_score(y_val, val_probs.argmax(1),average='macro',zero_division=1)
		val_loss = log_loss(y_val, val_probs)
		val_confs = val_probs.max(1)
		unl_probs = softmax(model(X_unl)).numpy()
		
		for v_i,v_conf in zip(val_indices,val_confs):
			sample_conf_mat[t,v_i] = v_conf
		
		preliminary_records['val_acc'].append(val_acc)
		preliminary_records['val_f1'].append(val_f1)
		preliminary_records['val_dist'].append(val_probs.mean(0))
		preliminary_records['ulb_dist'].append(unl_probs.mean(0))
		
		logger.info("In {}-th run, [best_val_acc,best_val_loss] = [{},{}].".format(t,round(val_acc,4),round(val_loss,4)))
		
	preliminary_records['sample_confs'] = sample_conf_mat.sum(0) / (sample_conf_mat!=0.0).astype('float32').sum(0)
	return preliminary_records

def obtain_outputs(preliminary_records, cali_acc_or_f1='f1', bias_lab_or_val='val'):
	T = len(preliminary_records['val_acc'])
	
	cali_dists = []
	for t in range(T):
		ulb_dist = preliminary_records['ulb_dist'][t]
		
		if cali_acc_or_f1=='acc':
			val_perf = preliminary_records['val_acc'][t]
		else:
			val_perf = preliminary_records['val_f1'][t]
		
		if bias_lab_or_val=='lab':
			smp_bias = preliminary_records['lab_dist']
		else:
			smp_bias = preliminary_records['val_dist'][t]
		
		cali_dists.append(class_calibration(ulb_dist, smp_bias, val_perf))
	
	p_l_conf = np.sort(preliminary_records['sample_confs']) # the confidences of labeled data
	c_u_cali = np.mean(cali_dists,0) # the output class distribution of unlabeled data
	return p_l_conf, c_u_cali

