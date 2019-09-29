from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from WarpImage import inverse_warp


def photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics,
                                    depth,  pose,
                                    rotation_mode='euler', padding_mode='zeros'):


    def one_scale(depth, mask):
        #assert( depth.size()[2:] == explainability_mask.size()[2:])
        '''
        explainability_mask is None or
        '''

        assert(pose.size(1) == len(ref_imgs))

        reconstruction_loss = 0
        b, _, h, w = depth.size()
        downscale = tgt_img.size(2)/h

        tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
        ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs]
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)

        warped_imgs = []
        diff_maps = []

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]

            ref_img_warped, valid_points = inverse_warp(ref_img, depth[:,0], current_pose,
                                                        intrinsics_scaled,
                                                        rotation_mode, padding_mode)
            diff = (tgt_img_scaled - ref_img_warped) * valid_points.unsqueeze(1).float()

            reconstruction_loss += diff.abs().mean()
            assert((reconstruction_loss == reconstruction_loss).item() == 1)

            warped_imgs.append(ref_img_warped[0])
            diff_maps.append(diff[0])

        return reconstruction_loss, warped_imgs, diff_maps

    warped_results, diff_results = [], []

    if type(depth) not in [list, tuple]:
        depth = [depth]

    total_loss = 0
    for d in zip(depth):
        loss, warped, diff = one_scale(d)
        total_loss += loss
        warped_results.append(warped)
        diff_results.append(diff)
    return total_loss, warped_results, diff_results