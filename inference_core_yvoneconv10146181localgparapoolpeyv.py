import torch

from inference_memory_bank46181localgparapoolperandyv import MemoryBank
from model.eval_networkoneconv10146181localgparapoolpe import STCN
from model.aggregate import aggregate

from util.tensor_util import pad_divide_by


class InferenceCore:
    def __init__(self, prop_net:STCN, images, num_objects, top_k=20, 
                    mem_every=5, include_last=False, req_frames=None):
        self.prop_net = prop_net
        self.mem_every = mem_every
        self.include_last = include_last

        self.req_frames = req_frames

        self.top_k = top_k

        # True dimensions
        t = images.shape[1]
        h, w = images.shape[-2:]

        # Pad each side to multiple of 16
        images, self.pad = pad_divide_by(images, 16)
        # Padded dimensions
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

        self.enabled_obj = []

        self.mem_banks = dict()

    def encode_key(self, idx):
        result = self.prop_net.encode_key(self.images[:,idx].cuda())
        return result

    def do_pass(self, key_k, key_v,key_vpe, idx, end_idx):
        closest_ti = end_idx

        K, CK, _, H, W = key_k.shape
        _, CV, _, _, _ = key_v.shape

        for i, oi in enumerate(self.enabled_obj):
            if oi not in self.mem_banks:
                self.mem_banks[oi] = MemoryBank(k=1, top_k=self.top_k)
            self.mem_banks[oi].add_memory(key_k, key_v[i:i+1], key_vpe[i:i+1])

        last_ti = idx

        this_range = range(idx+1, closest_ti)
        step = +1
        end = closest_ti - 1

        for ti in this_range: 
            is_mem_frame = (abs(ti-last_ti) >= self.mem_every)
            if (not is_mem_frame) and (not self.include_last) and (self.req_frames is not None) and (ti not in self.req_frames):
                continue

            k16, qv16, qf16, qf8, qf4 = self.encode_key(ti)

            out_mask = torch.cat([
                self.prop_net.segment_with_query(self.mem_banks[oi], qf8, qf4, k16, qv16)
            for oi in self.enabled_obj], 0)

            out_mask = aggregate(out_mask, keep_bg=True)
            self.prob[0,ti] = out_mask[0]
            for i, oi in enumerate(self.enabled_obj):
                self.prob[oi,ti] = out_mask[i+1]

            if ti != end:
                if self.include_last or is_mem_frame:
                    prev_value = self.prop_net.encode_value(self.images[:,ti].cuda(),self.images[:,ti+1].cuda(), qf16, out_mask[1:])
                    prev_valuepe = self.prop_net.encode_valuepe(self.images[:,ti].cuda(),self.images[:,ti+1].cuda(), qf16, out_mask[1:])

                    prev_key = k16.unsqueeze(2)
                    for i, oi in enumerate(self.enabled_obj):
                        self.mem_banks[oi].add_memory(prev_key, prev_value[i:i+1], prev_valuepe[i:i+1], is_temp=not is_mem_frame)

                    if is_mem_frame:
                        last_ti = ti

        return closest_ti

    def interact(self, mask, frame_idx, end_idx, obj_idx):
        mask, _ = pad_divide_by(mask.cuda(), 16)

        self.enabled_obj.extend(obj_idx)

        mask_regions = (mask[1:].sum(0) > 0.5)
        self.prob[:, frame_idx, mask_regions] = 0
        self.prob[obj_idx, frame_idx] = mask[obj_idx]

        self.prob[:, frame_idx] = aggregate(self.prob[1:, frame_idx], keep_bg=True)

        key_k, _, qf16, _, _ = self.encode_key(frame_idx)
        key_v = self.prop_net.encode_value(self.images[:,frame_idx].cuda(),self.images[:,frame_idx+1].cuda(), qf16, self.prob[self.enabled_obj,frame_idx].cuda())
        key_vpe = self.prop_net.encode_valuepe(self.images[:,frame_idx].cuda(),self.images[:,frame_idx+1].cuda(), qf16, self.prob[self.enabled_obj,frame_idx].cuda())

        key_k = key_k.unsqueeze(2)

        self.do_pass(key_k, key_v,key_vpe, frame_idx, end_idx)

