# BUS-stop
Accepted in ACL 2022 main conference 

"bus-stop-tensorflow" is a version for detailed analysis, so it is somewhat complicated.
"bus-stop-keras" is a easier version which is implemented with keras library.
Jupyter example in "bus-stop-keras" could help quick understanding of the overview of the bus-stop method.
Now, we are trying to implement py-torch version, which may be uploaded in July. 
The code and description will be continuously updated.

````
```
#You need to download pre-trained parameters. 
#See the below codes
import requests
URL = "https://cdn.huggingface.co/bert-base-uncased-pytorch_model.bin"
response = requests.get(URL)
open("./params/bert_base/pytorch_model.bin","wb").write(response.content)
```
````
