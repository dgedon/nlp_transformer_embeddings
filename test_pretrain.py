# Generic imports
import torch
import os
import argparse
from tqdm import tqdm
from warnings import warn
import json

# user defined inputs
from wsd.vocabulary import Vocabulary
from wsd.model.transformer_pretrain import MyTransformer


def predict_token(example, model, n):
    example_enc = voc.encode(example)
    # add mask token at the end
    example_enc[0].append(voc.get_mask_idx())
    # make tensor
    example_enc = torch.tensor(example_enc)

    output = model.predict_mask_token(example_enc)
    softmax = torch.nn.Softmax(dim=-1)
    tqdm.write("\n\n"+example[0])
    for _ in range(n):
        temp = softmax(output[0, -1, :]).max(-1)
        prob = temp[0].detach().item()
        idx = temp[1].detach().item()
        # remove highest value from output
        output[0, -1, idx] = -99999

        message = "Choice {}:\t'{}' with prob. {:2.2}%".format(n, itos[idx], prob * 100)
        tqdm.write(message)


if __name__ == '__main__':
    ###################################
    # Options and folders
    ###################################
    tqdm.write("Define test_pretrain.py settings...")

    # Experiment parameters
    config_parser = argparse.ArgumentParser(add_help=False)
    # Learning parameters

    # Model parameters
    config_parser.add_argument("--model_type", choices=['transformer_word', 'transformer_char'],
                               default='transformer_word',
                               help='model type.')
    args, rem_args = config_parser.parse_known_args()
    # System setting
    sys_parser = argparse.ArgumentParser(add_help=False)
    sys_parser.add_argument('--model_folder', type=str, default=os.getcwd() + '/wsd/server/pretrain_word/',
                            help='model folder.')

    settings, unk = sys_parser.parse_known_args(rem_args)
    #  Final parser is needed for generating help documentation
    parser = argparse.ArgumentParser(parents=[sys_parser, config_parser])
    _, unk = parser.parse_known_args(unk)
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")

    # Set device
    device = torch.device('cpu')


    tqdm.write("Done!")
    ###################################
    # Define the Model
    ###################################
    tqdm.write("Define model...")

    folder = settings.model_folder
    ckpt_pretrain_stage = torch.load(os.path.join(folder, 'pretrain_model.pth'),
                                     map_location=lambda storage, loc: storage)
    config_pretrain_stage = os.path.join(folder, 'pretrain_config.json')
    with open(config_pretrain_stage, 'r') as f:
        config_dict_pretrain_stage = json.load(f)
    tqdm.write("Found pretrained model!")
    # adapt some parameters
    args.max_voc_size = config_dict_pretrain_stage['max_voc_size']
    args.bag_of_chars = True

    if args.model_type in ['transformer_word']:
        try:
            temp = torch.load(os.path.join(settings["folder"], 'voc.pth'))
            max_char_voc_size = temp['max_voc_size']
            stoi = temp['stoi']
            itos = temp['itos']
        except:
            max_char_voc_size = args['max_char_voc_size']
            stoi = None
            itos = None
        max_voc_size = None
        voc = Vocabulary(max_voc_size=max_voc_size, stoi=stoi, itos=itos)

        class DummyClf:
            def __init__(self, voc):
                self.voc = voc

        # Load word model
        clf_input = DummyClf(voc)
        pretrained = MyTransformer(config_dict_pretrain_stage, clf_input)
        if ckpt_pretrain_stage is not None:
            pretrained.load_state_dict(ckpt_pretrain_stage['model'])

        model = pretrained.get_pretrained(train_words=True)

        # example sentence
        example = ['Today I will go to the store and ']
        predict_token(example, model, n=3)

        example = ['The sun ']
        predict_token(example, model, n=3)

        example = ['This is the beginning of a beautiful ']
        predict_token(example, model, n=3)
