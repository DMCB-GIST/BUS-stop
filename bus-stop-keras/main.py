"""
First author: HongSeok Choi (hongking9@gist.ac.kr)
Corresponding author: Hyunju Lee (hyunjulee@gist.ac.kr)
Code for Early Stopping Based on Unlabeled Samples in Text Classification.
"""

import os
import sys
import logging
import argparse
import numpy as np
import tensorflow as tf
import preliminary
import pt_modeler
import preprocessing as pp

from collections import deque
from pt_modeler import ConstructPtModeler
from huggingface_utils import MODELS
from tensorflow.keras.activations import softmax
from sklearn.utils import shuffle
from sklearn.metrics import log_loss, f1_score, accuracy_score
from scipy.spatial.distance import cosine,euclidean

logger = logging.getLogger('BUS-stop')
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s',"%H:%M:%S")
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--task", required=True)
	parser.add_argument("--data_path", default=None)
	parser.add_argument("--seed", type=int, default=0, help="Random seed for data shuffling")
	parser.add_argument("--pt_model", default="TFBertModel", help="Pre-trained model")
	parser.add_argument("--pt_model_checkpoint", default="./params/bert_base/", help="Model checkpoint to load pre-trained weights")
	parser.add_argument("--max_seq_length", type=int, default=64, help="Maximum sequence length")
	parser.add_argument("--word_freeze", type=bool, default=False, help="Word embedding freeze or not")
	parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for gradient updates")
	parser.add_argument("--drop_rate", type=float, default=0.2, help="dropout probability")
	parser.add_argument("--epochs", type=int, default=50, help="Maximum epoch")
	parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
	parser.add_argument("--patience", type=int, default=10, help="Patience for validation-based early stopping in preliminary stage")
	parser.add_argument("--val_ratio", type=float, default=0.5, help="The ratio of validation split in preliminary stage")
	parser.add_argument("--T", type=int, default=5, help="The number of resampling in preliminary stage")
	parser.add_argument("--n_base", type=int, default=5, help="The number of runs in preliminary stage")
	parser.add_argument("--n_que", type=int, default=5, help="Hyperparameter in main stage for BUS-stop")
	parser.add_argument("--verbose", type=int, default=0, choices=[0,1,2], help="Print type: 0 or 1 or 2")
	parser.add_argument("--cali_acc_or_f1", default='f1', choices=['acc','f1'], help="Hyperparameter for calibration: f1 or acc")
	parser.add_argument("--bias_lab_or_val", default='val', choices=['lab','val'], help="Hyperparameter for calibration: val or lab")
	parser.add_argument("--log_prefix", default='', help="Prefix of log file name")
	args = vars(parser.parse_args())

	#For logger
	fhandler = logging.FileHandler(filename='./logs/{}main.log'.format(args["log_prefix"]), mode='a')
	fhandler.setFormatter(formatter)
	fhandler.setLevel(logging.INFO)
	logger.addHandler(fhandler)

	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(formatter)
	consoleHandler.setLevel(logging.DEBUG)
	logger.addHandler(consoleHandler)

	logger_test = logging.getLogger('BUS-stop-test')
	logger_test.setLevel(logging.INFO)
	fhandler_test = logging.FileHandler(filename='./logs/{}main_test.log'.format(args["log_prefix"]), mode='a')
	fhandler_test.setFormatter(formatter)
	fhandler_test.setLevel(logging.INFO)
	logger_test.addHandler(fhandler_test)

	#Variables for preprocessing
	GLOBAL_SEED = args["seed"]
	pt_model = args["pt_model"]
	pt_model_checkpoint = args["pt_model_checkpoint"]
	task = args["task"]
	task_path = os.path.join('./data',task,args["data_path"]) # e.g. './data/SST-2/imbal/5'
	max_seq_length = args["max_seq_length"]

	#Common variables in the two stage
	word_freeze = args["word_freeze"]
	learning_rate = args["learning_rate"]
	drop_rate = args["drop_rate"]
	verbose = args["verbose"]
	epochs = args["epochs"]
	batch_size = args["batch_size"]

	#Preliminary stage variables
	patience = args["patience"]
	val_ratio = args["val_ratio"]
	T = args["T"]
	n_base = args["n_base"]
	cali_acc_or_f1 = args["cali_acc_or_f1"]
	bias_lab_or_val = args["bias_lab_or_val"]

	#Main stage variables
	n_que = args["n_que"]

	for indx, model in enumerate(MODELS):
		if model[0].__name__ == pt_model:
			TFModel, Tokenizer, Config = MODELS[indx]

	tokenizer = Tokenizer.from_pretrained(pt_model_checkpoint)

	devices = []
	for gpu_num in os.environ["CUDA_VISIBLE_DEVICES"].split(','):
		devices.append('/device:GPU:{}'.format(gpu_num))

	strategy = tf.distribute.MirroredStrategy(devices=devices)
	gpus = strategy.num_replicas_in_sync

	logger.info("***Logging start***")
	logger.info("task_path = {}".format(task_path))
	logger.info(args)
	logger.info("os.environ['CUDA_VISIBLE_DEVICES'] = {}".format(os.environ['CUDA_VISIBLE_DEVICES']))
	logger.info("devices = {}".format(devices))
	logger.info("Number of devices: {}".format(gpus))

	logger.info("*******************")
	logger.info("***Preprocessing***")
	logger.info("*******************")
	task = task.strip()
	Processor = pp.task_to_processor(task)
	processor = Processor(task, task_path, tokenizer, max_seq_length) 

	label_list = processor.get_label_list()
	lab_examples = processor.tsv_to_examples('labeled.tsv')
	tst_examples = processor.tsv_to_examples('test_with_gold.tsv') 
	unl_examples = processor.tsv_to_examples('unlabeled.tsv')

	X_lab,y_lab = processor.examples_to_features(lab_examples)
	X_tst,y_tst = processor.examples_to_features(tst_examples)
	X_unl,y_unl = processor.examples_to_features(unl_examples)

	lab_len, unl_len = len(lab_examples), len(unl_examples)
	num_labels = len(label_list)

	logger.info('Labeled//Test//Unlabeled matrix shape = {} // {} // {}'.format(
		X_lab['input_ids'].shape,X_tst['input_ids'].shape,X_unl['input_ids'].shape))

	for i in range(2):
		logger.info("***Train***")
		logger.info ("Example {}".format(i))
		logger.info ("Label {}".format(y_lab[i]))
		logger.info ("Token ids {}".format(X_lab["input_ids"][i]))
		logger.info ("Tokens {}".format(tokenizer.convert_ids_to_tokens(X_lab["input_ids"][i])))
		#logger.info ("Token type ids {}".format(X_lab["token_type_ids"][i]))
		logger.info ("Token mask {}".format(X_lab["attention_mask"][i]))

	for i in range(2):
		logger.info("***Test***")
		logger.info ("Example {}".format(i))
		logger.info ("Label {}".format(y_tst[i]))
		logger.info ("Token ids {}".format(X_tst["input_ids"][i]))
		logger.info ("Tokens {}".format(tokenizer.convert_ids_to_tokens(X_tst["input_ids"][i])))
		#logger.info ("Token type ids {}".format(X_tst["token_type_ids"][i]))
		logger.info ("Token mask {}".format(X_tst["attention_mask"][i]))

	for i in range(2):
		logger.info("***Unlabeled***")
		logger.info ("Example {}".format(i))
		logger.info ("Token ids {}".format(X_unl["input_ids"][i]))
		logger.info ("Tokens {}".format(tokenizer.convert_ids_to_tokens(X_unl["input_ids"][i])))
		#logger.info ("Token type ids {}".format(X_unl["token_type_ids"][i]))
		logger.info ("Token mask {}".format(X_unl["attention_mask"][i]))

	with strategy.scope():
		modeler = ConstructPtModeler(TFModel, Config, pt_model_checkpoint, max_seq_length, 
									 num_labels, dense_dropout_prob=drop_rate, word_freeze=word_freeze,
									 attention_probs_dropout_prob=drop_rate, hidden_dropout_prob=drop_rate)

	logger.info("***********************")
	logger.info("***Preliminary stage***")
	logger.info("***********************")
	preliminary_records = preliminary.run_stage(strategy, modeler, processor, lab_examples, X_unl, rand_seed=GLOBAL_SEED,
												epochs=epochs, patience=patience, batch_size=batch_size, 
												learning_rate=learning_rate, val_ratio=val_ratio, T=T, n_base=n_base, 
												verbose=verbose) #verbose=0/1/2 -> print silent/progress_bar/one_line_per_epoch. 
	p_l_conf, c_u_cali = preliminary.obtain_outputs(preliminary_records, cali_acc_or_f1=cali_acc_or_f1, bias_lab_or_val=bias_lab_or_val)

	#logger.info("preliminary_records = {}".format(preliminary_records))
	p_l_ = list(np.around(p_l_conf,4))
	logger.info("p_l_conf = [{},{},{},...,{},{},{}]".format(p_l_[0],p_l_[1],p_l_[2],p_l_[-3],p_l_[-2],p_l_[-1]))
	logger.info("class distribution of unlabeled data: pred {} -> cali {}".format(
		np.around(np.mean(preliminary_records['ulb_dist'],0),4), np.around(c_u_cali,4) ))

	logger.info("****************")
	logger.info("***Main stage***")
	logger.info("****************")
	with strategy.scope():
		model = modeler.build_model()
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08), 
					  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
					  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])

	steps_per_epoch = lab_len//batch_size
	queue = deque(n_que*[0], n_que)
	best_conf, n_pat = np.inf, 0
	rand_indices = np.arange(lab_len)
	for epoch in range(1,epochs+1):
		
		rand_indices = shuffle(rand_indices,random_state=GLOBAL_SEED)
		for step in range(steps_per_epoch):
			batch_indices = rand_indices[step*batch_size:(step+1)*batch_size]
			X_bat = {}
			for key in X_lab.keys():
				X_bat[key] = pp.select_by_index(X_lab[key], batch_indices)
			y_bat = pp.select_by_index(y_lab, batch_indices)
			model.train_on_batch(X_bat,y_bat)
		
		trn_loss,trn_acc = model.evaluate(X_lab, y_lab)
		tst_loss,tst_acc = model.evaluate(X_tst, y_tst) # 
		
		unl_probs = softmax(model(X_unl)).numpy()
		unl_confs = unl_probs.max(1)
		unl_dist = unl_probs.mean(0)
		
		_ids = np.arange(0,unl_len,unl_len/lab_len).astype('int32') # for downsampling
		s_conf = euclidean(unl_confs[_ids], p_l_conf)
		s_class = 1.-cosine(unl_dist, c_u_cali)
		#logger.info("Epoch {}, s_conf={}, s_class={}".format(epoch,round(s_conf,4),round(s_class,4)))
		logger.info("Epoch {}, s_conf={}, s_class={}, tst_acc={}, tst_loss={}".format(
				epoch, round(s_conf,4), round(s_class,4), round(tst_acc,4), round(tst_loss,4)))
		
		if s_conf < best_conf:
			n_pat = 0
			queue = deque(n_que*[0], n_que)
			best_conf = s_conf
		else:
			n_pat += 1 
		
		if n_pat < n_que:
			if s_class > max(queue):
				best_weights = model.get_weights()
				stop_epoch = epoch
			queue.append(s_class)
		else:
			break
	logger.info('***End training***')

	logger.info('***Load the model and Evaluate on test data***')
	logger.info("BUS-stop's stop_epoch = {}".format(stop_epoch))
	model.set_weights(best_weights)
	tst_loss,tst_acc = model.evaluate(X_tst, y_tst) # 
	logger.info('Final tst_acc : {}, tst_loss : {} \n'.format(round(tst_acc,4),round(tst_loss,4)))

	logger_test.info("{} -> tst_acc : {}, tst_loss : {}".format(task_path,round(tst_acc,4),round(tst_loss,4)))


