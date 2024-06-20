import random

import torch
from cryptos import *

from collections import namedtuple

BATCH = namedtuple('BATCH', ['src', 'trg'])


input_seq_len = 128
output_seq_len = 128

keys = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', '<sos>', '<pad>']


enc_voc_size = len(keys)
dec_voc_size = len(keys)


vocab_stoi = {}
vocab_itos = {}
for i, key in enumerate(keys):
    vocab_stoi[key] = i
    vocab_itos[i] = key


src_pad_idx = vocab_stoi['<pad>']
trg_pad_idx = vocab_stoi['<pad>']
trg_sos_idx = vocab_stoi['<sos>']


def text_to_idx(text):
    return [vocab_stoi[w] for w in text]


def idx_to_text(idx: torch.Tensor):
    l = idx.shape[0]
    return ''.join([vocab_itos[idx[i].item()] for i in range(l)])


class BCDataLoader:
    def __init__(self, total_size, batch_size, device):
        self.total_size = total_size
        self.batch_size = batch_size
        self.device = device

        self.batch_cnt = self.total_size // self.batch_size

    def text_to_idx(self, text):
        return [self.vocab_stoi[w] for w in text]

    def to_tensor(self, bathc_keys):
        src = []
        trg = []
        for key_src, key_trg in bathc_keys:
            src.append(text_to_idx(key_src))
            trg.append([trg_sos_idx] + text_to_idx(key_trg))

        return torch.tensor(src, device=self.device), torch.tensor(trg, device=self.device)

    def __len__(self):
        return self.batch_cnt * self.batch_size

    def __iter__(self):
        # bathc_keys = []
        # for _ in range(self.total_size):
        #     priv = random.randint(0, N)
        #     pub = fast_multiply(G, priv)
        #
        #     privkey = encode(priv, 16, 64)
        #     pubkey = encode(pub[0], 16, 64) + encode(pub[1], 16, 64)
        #
        #     bathc_keys.append((pubkey, privkey))
        #     if len(bathc_keys) == self.batch_size:
        #         src, trg = self.to_tensor(bathc_keys)
        #         bathc_keys = []
        #         yield BATCH(src=src, trg=trg)
        # ========================================================================================

        bathc_keys = []
        R = N // 2
        for _ in range(self.total_size):
            priv_src = random.randint(0, R)

            pub_src = fast_multiply(G, priv_src)
            pub_trg = fast_multiply(pub_src, 2)
            pubkey_src = encode(pub_src[0], 16, 64) + encode(pub_src[1], 16, 64)
            pubkey_trg = encode(pub_trg[0], 16, 64) + encode(pub_trg[1], 16, 64)

            bathc_keys.append((pubkey_src, pubkey_trg))
            if len(bathc_keys) == self.batch_size:
                src, trg = self.to_tensor(bathc_keys)
                bathc_keys = []
                yield BATCH(src=src, trg=trg)
        # ========================================================================================
        # for _ in range(self.batch_cnt):
        #     priv = random.randint(0, N)
        #     pub = fast_multiply(G, priv)
        #
        #     bathc_keys = []
        #     for _ in range(self.batch_size):
        #         privkey = encode(priv, 16, 64)
        #         pubkey = encode(pub[0], 16, 64) + encode(pub[1], 16, 64)
        #
        #         bathc_keys.append((pubkey, privkey))
        #         priv += 1
        #         pub = fast_add(pub, G)
        #
        #     src, trg = self.to_tensor(bathc_keys)
        #     yield BATCH(src=src, trg=trg)



if __name__ == '__main__':

    loader = BCDataLoader(8, 4, 0)

    for batch in loader:

        for i in range(4):
            print(idx_to_text(batch.trg[i]))
            print(idx_to_text(batch.src[i]))

        # print(batch.src.shape)
        # print(batch.trg.shape)
        # print(batch.trg)
        # print('=============================')

    # t = torch.ones(10, device=0)
    # print(t)
    # print(idx_to_text(t))