import torch

from inference_memory_bank46181localgparapoolperandfinaltiptestdev import MemoryBank
from model.eval_networkoneconv10146181localgparapoolperand import STCN
from model.aggregate import aggregate

from util.tensor_util import pad_divide_by

class InferenceCore:
    def __init__(self, prop_net:STCN, images, num_objects, top_k=20, mem_every=5, include_last=False):
        self.prop_net = prop_net
        self.mem_every = mem_every
        self.include_last = include_last
        t = images.shape[1]
        h, w = images.shape[-2:]

        images, self.pad = pad_divide_by(images, 16)
        nh, nw = images.shape[-2:]

        self.images = images
        self.device = 'cuda'

        self.k = num_objects

        self.prob = torch.zeros((self.k+1, t, 1, nh, nw), dtype=torch.float32, device=self.device)
        self.prob[0] = 1e-7

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh//16
        self.kw = self.nw//16

        self.mem_bank = MemoryBank(k=self.k, top_k=top_k)

    def encode_key(self, idx):
        result = self.prop_net.encode_key(self.images[:,idx].cuda())
        return result

    def do_pass(self, key_k, key_v,key_vpe, idx, end_idx):
        self.mem_bank.add_memory(key_k, key_v,key_vpe)

        closest_ti = end_idx
        this_range = range(idx+1, closest_ti)
        end = closest_ti - 1

        for ti in this_range:

            k16, qv16, qf16, qf8, qf4 = self.encode_key(ti)

            out_mask = self.prop_net.segment_with_query(self.mem_bank, qf8, qf4, k16, qv16)

            out_mask = aggregate(out_mask, keep_bg=True)
            self.prob[:,ti] = out_mask

            if ti != end:
                is_mem_frame = ((ti % self.mem_every) == 0)
                if self.include_last or is_mem_frame:
                    prev_value = self.prop_net.encode_value(self.images[:,ti].cuda(), self.images[:,ti+1].cuda(), qf16, out_mask[1:])
                    prev_valuepe = self.prop_net.encode_valuepe(self.images[:,ti].cuda(), self.images[:,ti+1].cuda(), qf16, out_mask[1:])

                    prev_key = k16.unsqueeze(2)
                    self.mem_bank.add_memory(prev_key, prev_value, prev_valuepe, is_temp=not is_mem_frame)

        return closest_ti

    def interact(self, mask, frame_idx, end_idx):
        mask, _ = pad_divide_by(mask.cuda(), 16)

        self.prob[:, frame_idx] = aggregate(mask, keep_bg=True)

        key_k, _, qf16, _, _ = self.encode_key(frame_idx)
        key_v = self.prop_net.encode_value(self.images[:,frame_idx].cuda(), self.images[:,frame_idx+1].cuda(),qf16, self.prob[1:,frame_idx].cuda())
        key_vpe = self.prop_net.encode_valuepe(self.images[:,frame_idx].cuda(), self.images[:,frame_idx+1].cuda(),qf16, self.prob[1:,frame_idx].cuda())

        key_k = key_k.unsqueeze(2)
        self.do_pass(key_k, key_v,key_vpe, frame_idx, end_idx)

