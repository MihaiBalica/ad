import sys
sys.path.append('core')

import argparse
import os
import time
import numpy as np
import torch

from sparsenet import SparseNet
# from sparsenet import SparseNetEighth

import datasets
from utils import frame_utils

from utils.utils import InputPadder, forward_interpolate



@torch.no_grad()
def create_kitti_submission(model, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

        flow_pr = model.module(image1, image2)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)




@torch.no_grad()
def validate_kitti(model, iters=6):
    """ Peform validation using the KITTI-2015 (train) split """
    output_path='/content/datasets/KITTI/image_02/flow'
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        # image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1, image2 = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_pr = model(image1, image2, iters=iters, test_mode=True)
        # flow = padder.unpad(flow_pr[0]).cpu()
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
        output_filename = os.path.join(output_path, str(val_id)) + ".flo"
        print(output_filename)
        # frame_utils.writeFlowKITTI(output_filename, flow)
        frame_utils.writeFlow(output_filename, flow)

        # epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        # mag = torch.sum(flow_gt**2, dim=0).sqrt()

        # epe = epe.view(-1)
        # mag = mag.view(-1)
        # val = valid_gt.view(-1) >= 0.5

        # out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        # epe_list.append(epe[val].mean().item())
        # out_list.append(out[val].cpu().numpy())

    # epe_list = np.array(epe_list)
    # out_list = np.concatenate(out_list)

    # epe = np.mean(epe_list)
    # f1 = 100 * np.mean(out_list)

    # print("Validation KITTI: %f, %f" % (epe, f1))
    # return {'kitti_epe': epe, 'kitti_f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--num_k', type=int, default=8,
                        help='number of hypotheses to compute for knn Faiss')
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')

    args = parser.parse_args()

    model = torch.nn.DataParallel(SparseNet(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module, iters=12)

        elif args.dataset == 'sintel':
            validate_sintel(model.module, iters=32)

        elif args.dataset == 'kitti':
            validate_kitti(model.module, iters=24)
