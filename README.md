# L101 project - Natural Language Inference

## Prerequisites
The project heavily utilises GPU parallelism, so it is assumed that you have
PyTorch with CUDA functionality installed and working. 

Download and place the data in the `.data` folder. Download and place
pre-trained vector embeddings in the `.vector_cache` folder. Create
`.serialization data` folder where training data, together with the best/latest
trained models will be placed.

## Usage
Currently, there is no GUI or command line interface. If you want to change the
architecture/word embeddings, you have to modify the code. Thankfully, I have
left comments pointing where to edit and examples how to edit.

Note that, the class implementing Rocktaschel et al's attention networks
`RocktaschelEtAlAttention` has `word_by_word` boolean flag in the constructor,
with which you can control what attention to use. Similarly,
`RocktaschelEtAlConditionalEncoding` has `use_fastgrnn` flag to switch between
gating mechanism.
