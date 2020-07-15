import torch.nn as nn

class WordSelModel(nn.Module):
    """
    Selects the correct word from the transformer word embedding output
    """

    def __init__(self, ):
        super(WordSelModel, self).__init__()

    def forward(self, src):
        src, word_pos = src

        src1 = src.transpose(0, 1)

        idx = word_pos.view(-1, 1).unsqueeze(2).expand(-1, 1, src1.shape[2]).to(device=src.device)
        out = src1.gather(1, idx)
        output = out.squeeze()

        return output
