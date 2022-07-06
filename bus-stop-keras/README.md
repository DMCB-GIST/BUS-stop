# Keras-version

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
