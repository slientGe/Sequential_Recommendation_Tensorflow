# Sequential_Recommendation_Tensorflow
In this repository, a number of sequential recommendation models are implemented using Python and Tensorflow. 
The implemented models cover common sequential recommendation algorithms (session based ). We implement the code in the paper in a concise way, including how to construct samples and training, to help readers better understand the paper's ideas.


# Algorithms Implemented

So far, we have implemented these models, covering deep learning and traditional methods. Follow up to continue to update。

| model   | paper    | methods|
| ------ | ------ | ------ | 
| AttRec | Next Item Recommendation with Self-Attention   |  self-attention |
| Caser |  Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding  |  CNN     |     
| GRU4Rec | Session-based Recommendations with Recurrent Neural Networks   |  GRU     |   
| FPMC | Factorizing Personalized Markov Chains for Next-Basket Recommendation  |  MF+MC    |    
| TransRec | Translation-based Recommendation |  MF      | 
| SASRec| Self-Attentive Sequential Recommendation |transfomer|

and so on.


# Usage

 To use the code, enter the models directory and execute run_Model.py
such as:
``` bash
cd models/AttRec
python run_Attrec.py
```
Note: Due to the different sample construction methods and experimental methods of different algorithms, we generate independent codes for each algorithm.

    
# Requirements
* Tensorflow 1.1+
* Python 3.6+, 
* numpy
* pandas

# ToDo List
* More models
* Code refactoring
* Support tf.data.datasets and tf.estimator


