
IMG_CHAN = 3
def inv_rigid_transformation(rot_mat_batch, trans_batch):
    inv_rot_mat_batch = rot_mat_batch.transpose(1,2)
    inv_trans_batch = -inv_rot_mat_batch.bmm(trans_batch.unsqueeze(-1)).squeeze(-1)
    return inv_rot_mat_batch, inv_trans_batch

def compute_SSIM(img0, mu0, sigma0, img1, mu1, sigma1):
    # img0_img1_pad = torch.nn.ReplicationPad2d(1)(img0 * img1)
    img0_img1_pad = img0*img1
    sigma01 = AvgPool2d(kernel_size=3, stride=1, padding=0)(img0_img1_pad) - mu0*mu1
    # C1 = .01 ** 2
    # C2 = .03 ** 2
    C1 = .001
    C2 = .009

    ssim_n = (2*mu0*mu1 + C1) * (2*sigma01 + C2)
    ssim_d = (mu0**2 + mu1**2 + C1) * (sigma0 + sigma1 + C2)
    ssim = ssim_n / ssim_d
    return ((1-ssim)*.5).clamp(0, 1)

def compute_img_stats(img):
    # img_pad = torch.nn.ReplicationPad2d(1)(img)
    img_pad = img
    mu = AvgPool2d(kernel_size=3, stride=1, padding=0)(img_pad)
    sigma = AvgPool2d(kernel_size=3, stride=1, padding=0)(img_pad**2) - mu**2
    return mu, sigma

def compute_phtometric_loss(self, ref_frames_pyramid, src_frames_pyramid, ref_inv_depth_pyramid, src_inv_depth_pyramid,
                            rot_mat_batch, trans_batch,
                            use_ssim=True, levels=None,
                            ref_expl_mask_pyramid=None,
                            src_expl_mask_pyramid=None):
    bundle_size = rot_mat_batch.size(0) + 1
    inv_rot_mat_batch, inv_trans_batch = inv_rigid_transformation(rot_mat_batch, trans_batch)
    src_pyramid = []
    ref_pyramid = []
    depth_pyramid = []
    if levels is None:
        levels = range(self.pyramid_layer_num)



    # for level_idx in range(len(ref_frames_pyramid)):
    for level_idx in levels:
        # for level_idx in range(3):
        ref_frame = ref_frames_pyramid[level_idx].unsqueeze(0).repeat(bundle_size - 1, 1, 1, 1)
        src_frame = src_frames_pyramid[level_idx]
        ref_depth = ref_inv_depth_pyramid[level_idx].unsqueeze(0).repeat(bundle_size - 1, 1, 1)
        src_depth = src_inv_depth_pyramid[level_idx]
        # print(src_depth.size())
        ref_pyramid.append(torch.cat((ref_frame,
                                      src_frame), 0) / 127.5)
        src_pyramid.append(torch.cat((src_frame,
                                      ref_frame), 0) / 127.5)
        depth_pyramid.append(torch.cat((ref_depth,
                                        src_depth), 0))

    rot_mat = torch.cat((rot_mat_batch,
                         inv_rot_mat_batch), 0)
    trans = torch.cat((trans_batch,
                       inv_trans_batch), 0)

    loss = 0

    frames_warp_pyramid = []
    ref_frame_warp_pyramid = []

    # for level_idx in range(len(ref_pyramid)):
    # for level_idx in range(3):
    for level_idx in levels:
        # print(depth_pyramid[level_idx].size())
        _, h, w = depth_pyramid[level_idx].size()
        warp_img, in_view_mask = self.warp_batch_func(
            src_pyramid[level_idx],
            depth_pyramid[level_idx],
            level_idx, rot_mat, trans)
        warp_img = warp_img.view((bundle_size - 1) * 2, IMG_CHAN, h, w)

        rgb_loss = ((ref_pyramid[level_idx] - warp_img).abs() )
        if use_ssim and level_idx < 1:
            # print("compute ssim loss------")
            warp_mu, warp_sigma = compute_img_stats(warp_img)
            ref_mu, ref_sigma = compute_img_stats(ref_pyramid[level_idx])
            ssim = compute_SSIM(ref_pyramid[level_idx],
                                ref_mu,
                                ref_sigma,
                                warp_img,
                                warp_mu,
                                warp_sigma)
            ssim_loss = (ssim * mask_expand[:, :, 1:-1, 1:-1]).mean()
            loss += .85 * ssim_loss + .15 * rgb_loss
        else:
            loss += rgb_loss

    #     frames_warp_pyramid.append(warp_img*127.5)
    #     ref_frame_warp_pyramid.append(ref_pyramid[level_idx]*127.5)
    #
    # return loss, frames_warp_pyramid, ref_frame_warp_pyramid
    return loss