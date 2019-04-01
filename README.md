# The Underworlds KB [Work In Progress]
This repository contains an example of KB for Underworlds made on top of Oro associated with a NL to SPAQRL translater

----
This component allow to query the RDF server in natural language by exploiting LSTM Encoder-Decoder architecture to translate the NL query into a SPARQL query that will then be processed by the Ontology reasoner.

We assume that you already have Underworlds installed, if not follow this [guide](https://github.com/underworlds-robot/uwds/blob/master/QUICKSTART.md).

# Installation instructions
### Ontology related
First, clone the repo and download the KB with :

```shell
git clone https://github.com/underworlds-robot/uwds_knowledge_base.git
cd uwds_knowledge_base
./oro_install.sh -i
```
### Deep learning related

#### Without GPU (recommended for beginners)
Download TensorFlow and Keras :
```shell
pip install --user tensorflow
pip install --user keras
```
#### With GPU
Requires a CUDA compatible GPU, see list [here](https://developer.nvidia.com/cuda-gpus)

Install CUDA for your GPU by following this [guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

Then download TensorFlow and Keras :
```shell
pip install --user tensorflow-gpu
pip install --user keras
```
You can optionally download the dataset for learning by executing the `./download_data.sh` script

# Launch instructions


```shell
roslaunch uwds_knowledge_base uwds_knowledge_base.launch
```
