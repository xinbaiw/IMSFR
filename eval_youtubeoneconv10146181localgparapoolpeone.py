


import os
from os import path
from argparse import ArgumentParser
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.eval_networkoneconv10146181localgparapoolpe import STCN
from dataset.yv_test_dataset import YouTubeVOSTestDataset
from util.tensor_util import unpad
from inference_core_yvoneconv10146181localgparapoolpeyv import InferenceCore

from progressbar import progressbar

"""
Arguments loading
"""
parser = ArgumentParser()

parser.add_argument('--model', default='/..')

parser.add_argument('--yv_path', default='/../Youtube-VOS')


parser.add_argument('--output_all', help=
"""
We will output all the frames if this is set to true.
Otherwise only a subset will be outputted, as determined by meta.json to save disk space.
For ensemble, all the sources must have this setting unified.
""", action='store_true')
parser.add_argument('--output', default='/..')

parser.add_argument('--split', help='valid/test', default='valid')
parser.add_argument('--top', type=int, default=20)
# parser.add_argument('--top', type=int, default=17)

parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=3, type=int)

parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
args = parser.parse_args()

yv_path = args.yv_path
out_path = args.output

# Simple setup
os.makedirs(out_path, exist_ok=True)
palette = Image.open(path.expanduser(yv_path + '/valid/Annotations/0a49f5265b/00000.png')).getpalette()

torch.autograd.set_grad_enabled(False)

# Load the json if we have to
if not args.output_all:
    with open(path.join(yv_path, args.split, 'meta.json')) as f:
        meta = json.load(f)['videos']

test_dataset = YouTubeVOSTestDataset(data_root=yv_path, split=args.split)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

prop_saved = torch.load(args.model)
top_k = args.top
prop_model = STCN().cuda().eval()
prop_model.load_state_dict(prop_saved)

for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

    with torch.cuda.amp.autocast(enabled=args.amp):
        rgb = data['rgb']
        msk = data['gt'][0]
        info = data['info']
        name = info['name'][0]
        num_objects = len(info['labels'][0])
        gt_obj = info['gt_obj']
        size = info['size']

        req_frames = None
        if not args.output_all:
            req_frames = []
            objects = meta[name]['objects']
            for key, value in objects.items():
                req_frames.extend(value['frames'])

            req_frames_names = set(req_frames)
            req_frames = []
            for fi in range(rgb.shape[1]):
                frame_name = info['frames'][fi][0][:-4]
                if frame_name in req_frames_names:
                    req_frames.append(fi)
            req_frames = sorted(req_frames)

        frames_with_gt = sorted(list(gt_obj.keys()))

        processor = InferenceCore(prop_model, rgb, num_objects=num_objects, top_k=top_k, 
                                    mem_every=args.mem_every, include_last=args.include_last, 
                                    req_frames=req_frames)

        min_idx = 99999
        for i, frame_idx in enumerate(frames_with_gt):
            min_idx = min(frame_idx, min_idx)
            obj_idx = gt_obj[frame_idx][0].tolist()
            obj_idx = [info['label_convert'][o].item() for o in obj_idx]

            with_bg_msk = torch.cat([
                1 - torch.sum(msk[:,frame_idx], dim=0, keepdim=True),
                msk[:,frame_idx],
            ], 0).cuda()

            if i == len(frames_with_gt) - 1:
                processor.interact(with_bg_msk, frame_idx, rgb.shape[1], obj_idx)
            else:
                processor.interact(with_bg_msk, frame_idx, frames_with_gt[i+1]+1, obj_idx)

        out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')

        for ti in range(processor.t):
            prob = unpad(processor.prob[:,ti], processor.pad)
            prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
            out_masks[ti] = torch.argmax(prob, dim=0)

        out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

        idx_masks = np.zeros_like(out_masks)
        for i in range(1, num_objects+1):
            backward_idx = info['label_backward'][i].item()
            idx_masks[out_masks==i] = backward_idx
        
        # Save the results
        this_out_path = path.join(out_path, 'Annotations', name)
        os.makedirs(this_out_path, exist_ok=True)
        for f in range(idx_masks.shape[0]):
            if f >= min_idx:
                if args.output_all or (f in req_frames):
                    img_E = Image.fromarray(idx_masks[f])
                    img_E.putpalette(palette)
                    img_E.save(os.path.join(this_out_path, info['frames'][f][0].replace('.jpg','.png')))

        del rgb
        del msk
        del processor
