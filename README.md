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

Download the required package into a virtual environment with :

```shell
virtualenv uwds_kb
source uwds_kb/bin/activate
pip install OpenNMT-tf
```

Note : The virtual environment will be activated by the r

# Launch instructions

```shell
roslaunch uwds_knowledge_base uwds_knowledge_base.launch
```
