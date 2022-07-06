# BUS-stop
This code for 'Early Stopping Based on Unlabeled Samples in Text Classification', which is accepted in ACL 2022 main conference.  
You can refer to [https://aclanthology.org/2022.acl-long.52/](https://aclanthology.org/2022.acl-long.52/).

"bus-stop-tensorflow" is a version for detailed analysis, so it is somewhat complicated.  
We recommend to use "bus-stop-keras", which is a easier version implemented with keras (tensorflow) library.  
Jupyter example 'run-cell-by-cell.ipynb' in "bus-stop-keras" could help quick understanding about overall algorithm.  
Now, we are trying to implement pytorch version, which may be uploaded in July or August.  
The code and description will be continuously updated.  

First, you need to download pre-trained parameters. 
````
# For the tensorflow version, 
# Refer to "https://github.com/google-research/bert" to download the pretrained parameters

# For the keras version,
# You can download pre-trained parameters as below.
# The file name must be "pytorch_model.bin".

import requests
URL = "https://cdn.huggingface.co/bert-base-uncased-pytorch_model.bin"
response = requests.get(URL)
open("./params/bert_base/pytorch_model.bin","wb").write(response.content)
````

# Keras-version ReadMe
We used Anaconda, and follows the below commands to implement the virtual enviroment.
````
conda create --name env_name python=3.6
conda activate env_name 
conda install -c anaconda tensorflow-gpu=2.2.0
conda install -c conda-forge transformers=3.3.1
conda install -c anaconda scikit-learn
conda install -c conda-forge dataclasses=0.7
conda install -c conda-forge sentencepiece=0.1.91
````

"setup_experiments.py" can generate five sample sets for each balanced and imbalanced setting (K=50) of SST-2 dataset. (Refer to the paper for detail about the experiments)  
````
python setup_experiments.py
````

After that, you can experiments with these settings.  
For example, if you want to use gpu0 and experiment with the balanced classification, then the command is as below. 
````
for t in {0..4}; do CUDA_VISIBLE_DEVICES=0 python main.py --task SST-2 --data_path bal/$t --seed $t --log_prefix bal_ --word_freeze True; done
````

If you want to use mult-gpu (0 and 1) and experiment with the imbalanced classification, then the command is as below.
````
for t in {0..4}; do CUDA_VISIBLE_DEVICES=0,1 python main.py --task SST-2 --data_path imbal/$t --seed $t --log_prefix imbal_ --word_freeze True; done
````
