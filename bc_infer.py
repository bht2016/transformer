import math
import random
import time
import torch

from cryptos import *


from bc_train import model, device
from bc_hex_dataloader import text_to_idx, trg_sos_idx, idx_to_text, output_seq_len


def infer(model, pubk):

    idx = text_to_idx(pubk[2:])
    input = torch.tensor(idx, device=device)
    input = input.view(1, -1)
    # print(input.shape)

    output = torch.tensor([trg_sos_idx] + [0 for _ in range(output_seq_len-1)], device=device)
    output = output.view(1, -1)
    # print(output.shape)

    with torch.no_grad():

        input_mask = model.make_src_mask(input)
        enc_src = model.encoder(input, input_mask)

        output = output[:, :-1]
        output_mask = model.make_trg_mask(output)
        for _ in range(output_seq_len-1):
            output = model.decoder(output, enc_src, output_mask, input_mask)
            output = output[0].max(dim=1)[1]
            output = output.contiguous().view(1, -1)

        output = output.contiguous().view(-1)
        return idx_to_text(output)


if __name__ == '__main__':

    model.load_state_dict(torch.load('saved/model.pt'))

    # priv = random.randint(0, N)
    #
    # pub = fast_multiply(G, priv)
    #
    # pubkey = encode_pubkey(pub, 'hex')
    #
    # pre_privkey = infer(model, pubkey)
    #
    # print('pred: ', pre_privkey)
    # print('real: ', encode(priv, 16, 64))

    priv = random.randint(0, N)
    pub = fast_multiply(G, priv)
    pubkey = encode_pubkey(pub, 'hex')
    pre_privkey = infer(model, pubkey)

    priv2 = priv // 2
    pub2 = fast_multiply(G, priv2)
    real_pubkey = encode_pubkey(pub2, 'hex')[2:]

    print(priv % 2)
    print('pred: ', pre_privkey)
    print('real: ', real_pubkey)

    indics = []
    for i in range(len(pre_privkey)):
        if pre_privkey[i] == real_pubkey[i]:
            indics.append(i)

    print(indics)
