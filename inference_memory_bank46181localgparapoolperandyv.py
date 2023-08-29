import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import random

def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = values.exp_()

    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    x.zero_().scatter_(1, indices, x_exp.type(x.dtype))  # B * THW * HW

    return x


class MemoryBank:
    def __init__(self, k, top_k=20):
        self.top_k = top_k

        self.CK = None
        self.CV = None

        self.mem_k = None
        self.mem_v = None
        self.premem_k = None
        self.premem_v = None
        self.firstmem_k = None
        self.firstmem_v = None

        self.num_objects = k

    def _global_matching(self, mk, qk):
        B, CK, NE = mk.shape
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk
        affinity = (2 * ab - a_sq) / math.sqrt(CK)  # B, NE, HW
        affinity = softmax_w_top(affinity, top=self.top_k)  # B, NE, HW

        return affinity

    def _global_matching1(self, mk, qk):
        B, CK, NE = mk.shape
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk
        affinity = (2 * ab - a_sq) / math.sqrt(CK)  # B, NE, HW
        affinity = F.softmax(affinity, dim=1)
        return affinity

    def _readout(self, affinity, mv):

        return torch.bmm(mv, affinity)

    def match_memory(self, qk):
        k = self.num_objects
        _, _, h, w = qk.shape

        qk = qk.flatten(start_dim=2)

        if self.temp_k is not None:
            mkj = torch.cat([self.mem_k, self.temp_k], 2)
            mvj = torch.cat([self.mem_v, self.temp_v], 2)
            firstmem_k = self.firstmem_k
            firstmem_v = self.firstmem_v
            premem_k = self.temp_k
            premem_v = self.premem_v

        else:
            mk = self.mem_k
            mv = self.mem_v

            Bj, CKj, Tj, Hj, Wj = mk.shape

            if Tj <=10:
                mk = mk.flatten(start_dim=2)

                mkj = mk
                mv = mv.flatten(start_dim=2)

                mvj = mv

            else:
                list = (range(0,mk.shape[2]-1))

                target = random.sample(list,int(mk.shape[2] *0.6))
                target.sort()

                mkj = mk[:,:,target]
                mkj = torch.cat([mk[:,:,0:2],mkj],2)
                mkj = torch.cat([mk[:,:,-5:-1],mkj],2)
                # mkj = torch.cat([mk[:,:,-1].unsqueeze(2),mkj],2)

                mkj = mkj.flatten(start_dim=2)

                mvj = mv[:,:,target]
                mvj = torch.cat([mv[:,:,0:2],mvj],2)
                mvj = torch.cat([mv[:,:,-5:-1],mvj],2)
                # mvj = torch.cat([mv[:,:,-1].unsqueeze(2),mvj],2)

                mvj = mvj.flatten(start_dim=2)

            firstmem_k = self.firstmem_k
            firstmem_v = self.firstmem_v
            firstmem_k = firstmem_k.flatten(start_dim=2)
            firstmem_v = firstmem_v.flatten(start_dim=2)

            premem_k = self.premem_k
            premem_v = self.premem_v
            premem_k = premem_k.flatten(start_dim=2)

        affinitymemory = self._global_matching(mkj, qk)
        affinityfirst = self._global_matching1(firstmem_k, qk)
        affinitylocal = self._global_matching1(premem_k, qk)
        readout_mem = self._readout(affinitymemory.expand(k, -1, -1), mvj)
        firstreadout_mem = self._readout(affinityfirst.expand(k, -1, -1), firstmem_v)

        localreadout_mem = self._readout(affinitylocal.expand(k, -1, -1), premem_v)

        readout_mem = readout_mem.view(k, self.CV, h, w)
        firstreadout_mem = firstreadout_mem.view(k, self.CV, h, w)
        localreadout_mem = localreadout_mem.view(k, self.CV, h, w)
        return readout_mem,firstreadout_mem,localreadout_mem

    def add_memory(self, key, value, valuepe, is_temp=False):

        self.temp_k = None
        self.temp_v = None
        valuepe = valuepe.flatten(start_dim=2)

        if self.mem_k is None:
            self.mem_k = key
            self.mem_v = value
            self.premem_k = key
            self.premem_v = valuepe
            self.firstmem_k = key
            self.firstmem_v = value

            self.CK = key.shape[1]
            self.CV = value.shape[1]
        else:
            if is_temp:
                self.temp_k = key
                self.temp_v = value

            else:
                self.mem_k = torch.cat([self.mem_k, key], 2)
                self.mem_v = torch.cat([self.mem_v, value], 2)
                self.premem_k = key
                self.premem_v = valuepe