# Evaluating Transformers as Embedding Layers

This is the final course project for the [WASP](https://wasp-sweden.org/) course on natural language processing, see [here](https://liu-nlp.github.io/dl4nlp/) for course website. The project employs transformers as embedding layers on the word sense disambiguation (WSD) task and compares different level of text abstraction, i.e. character level embedding, word level embedding and a combination of both. We consider the WSD task as a document classification problem without considering the exacat word position. For more details, see the project report available at [/doc](https://github.com/dgedon/nlp_transformer_embeddings/tree/master/doc).

The code is completely self written and only based on the native PyTorch transformer implementation, see [here](https://pytorch.org/docs/master/nn.html?highlight=nn%20transformer#transformer-layers) for the documentation. We employ a [BERT](https://arxiv.org/pdf/1810.04805.pdf) training objective by masking out 15\% of the input tokens and predicting them. We consider robust improvements by the [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf) model as training objective.

The training procedure is a two step procedure. First, a language model is pretrained. For this run
```
python pretrain.py --cuda 
```
For distinguishing between word or character level transformer use the option `--transformer_type [words, chars]`.

For training a simple embedding classifciation run
```
python train.py --cuda
```
with the option `--model_type ['simple_word', 'simple_char', 'simple_word_char']`. 

For using the pretrained transformer embedding and finetuning it on the WSD task run
```
python train.py --cuda --folder /PATH/TO/FOLDER/WITH/PRETRAINED/MODEL --model_type ['transformer_word', 'transformer_char', 'transformer_word_char']
```
Note that for a combined transformer word + character level model that `--folder` is for the word level folder and `--folder_model2` is for the character level folder.
