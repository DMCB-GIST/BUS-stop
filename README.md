# BUS-stop-code
This code for 'Early Stopping Based on Unlabeled Samples in Text Classification', which is accepted in ACL 2022 main conference.  
You can refer to [https://aclanthology.org/2022.acl-long.52/](https://aclanthology.org/2022.acl-long.52/).  

"bus-stop-tensorflow" is a version for detailed analysis, so it is somewhat complicated.  
We recommend to use "bus-stop-keras", which is a easier version implemented with keras (tensorflow) library.  
Jupyter example 'run-cell-by-cell.ipynb' in "bus-stop-keras" could help quick understanding about overall algorithm.  
We can also develop new versions upon request. The code and description may be updated later.  
(* This code is for non-commercial use.)

# Performance 
* **Accuracy in Balanced Classification**  

|                 | SST-2     | IMDB      | Elec      | AG-news   | DBpedia   | Average   |
|-----------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Val-split(25)   | 0.775     | 0.746     | 0.781     | 0.846     | 0.982     | 0.826     |
| Val-add(25)*    | 0.819     | 0.824     | 0.842     | **0.867** | **0.986** | 0.868     |
| EB-criterion    | 0.826     | **0.833** | 0.843     | 0.861     | **0.986** | 0.869     |
| LID             | 0.794     | 0.761     | 0.815     | 0.859     | 0.971     | 0.840     |
| **BUS-stop (ours)** | **0.831** | 0.828     | **0.848** | 0.865     | **0.986** | **0.872** |

All methods used 50 samples per class, except for the Val-add(25)*.  
*Val-add(25) uses 25 additional labeled samples for early stopping, which has an unfair advantage.

* **Accuracy in Imbalanced Classification**  

|                 | SST-2     | IMDB      | Elec      | Average   |
|-----------------|-----------|-----------|-----------|-----------|
| Val-split(25)   | 0.788     | 0.732     | 0.783     | 0.768     |
| Val-add(25)*    | 0.823     | 0.820     | 0.837     | 0.827     |
| EB-criterion    | 0.846     | 0.810     | 0.839     | 0.832     |
| LID             | 0.750     | 0.712     | 0.780     | 0.747     |
| **BUS-stop (ours)** | **0.860** | **0.849** | **0.876** | **0.861** |

The test class distribution was adjusted to 2:8 (neg:pos) and other settings are the same as the balanced classification.  
BUS-stop shows better performance peculiarly in imbalanced and low resource settings.  

* **Examples**  

![fig5](https://github.com/DMCB-GIST/BUS-stop/blob/main/ppts/fig5.png)  
BUS-stop is a combination of conf-sim and class-sim. The red vertical line denotes the best model selected by the BUS-stop.  
The loss and conf-sim were scaled between 0.25-0.75 for easy comparison.  
The BUS-stop enables fine-stopping, which skillfully avoids the points where the performance is decreased by fluctuations.

# How to run <sub><sup>(for Keras version)</sup></sub>
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
