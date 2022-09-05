"""
First author: HongSeok Choi (hongking9@gist.ac.kr)
Corresponding author: Hyunju Lee (hyunjulee@gist.ac.kr)
Code for Early Stopping Based on Unlabeled Samples in Text Classification.
"""
import requests

URL = "https://cdn.huggingface.co/bert-base-uncased-pytorch_model.bin"
response = requests.get(URL)
open("./pytorch_model.bin","wb").write(response.content)