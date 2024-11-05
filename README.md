# MiGNN2CDS


1. Description
   
Circular RNAs (circRNAs) can regulate microRNA activity and are related to various diseases, such as cancer. Functional research on circRNAs is the focus of scientific research. Accurate identification of circRNAs is important for gaining insight into their functions.

We developed a novel framework, MiGNN2CDS, for predicting circRNA and drug sensitivity association. MiGNN2CDS uses four different feature encoding schemes and adopts a multilayer convolutional neural network and bidirectional long short-term memory network to learn high-order feature representation and make circRNA predictions. 

2. Availability

## Datasets and source code are available at:https://github.com/nmt315320/MiGNN2CDS.git.

association.csv:circRNA-drug sensitivity association dataset.
drug_str_sim.csv:circRNA similarity feature dataset.
gene_seq_sim.csv:drug similarity dataset.
drug_pubchem_map.csv: a comparison table of drug ids and names in the association matrix. The first column is the id in the association matrix, the second column is the corresponding drug name, and the third column is the drug id in pubchem.
gene_map.csv: a comparison table of circRNA ids and names in the association matrix. The first column is the id in the association matrix, the second column is the id of the Gene corresponding to circRNA in the ncbi gene database, and the third column is its name.

## Local running

## Environment

Before running, please make sure the following packages are installed in Python environment:

gensim==3.4.0

pysam

pigwig

pandas==1.1.3

tensorflow==2.3.0

python==3.7.3

biopython==1.7.8

numpy==1.19.2

For convenience, we strongly recommended users to install the Anaconda Python 3.7.3 (or above) in your local computer.

3. Running

## k-fold Cross Validation
    python main.py -da {DATASET} -sp {SAVED PATH}
    Main arguments:
        -da: B-dataset 
        -ag: Aggregation method for bag embedding [sum, mean, Linear, BiTrans]
        -nl: The number of HeteroGCN layer
        -tk: The topk similarities in heterogeneous network construction
        -k : The topk filtering in instance predictor
        -hf: The dimension of hidden feature
        -ep: The number of epoches
        -bs: Batch size
        -lr: Learning rate
        -dp: Dropout rate
    For more arguments, please see args.py
Note: please see the optimal hyperparameter settings for each benchmark dataset, and other support information in 'supplementary materials.docx'.  

## Model Intepretebility
Use the ``model_intepret.ipynb`` to easily generate topk most important **meta-path instances** for given drug-disease pair (require **pre-trained model** first). 

3.4 Output explaining

The output file (in ".csv" format) can be found in results folder, which including sequence number, sequence_id, predicted probability and pedicted result.


