import torch

from imageio import imread, imsave
from scipy.misc import imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import SharedEncoder, DepthDecoder
from utils import tensor2array

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained-dispnet-enc", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--pretrained-dispnet-dec", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return


    disp_net_enc = SharedEncoder.SharedEncoderMain().double().to(device)
    weights = torch.load(args.pretrained_dispnet_enc, map_location=lambda storage, loc: storage)

    disp_net_enc.load_state_dict(weights)
    disp_net_enc.eval()

    disp_net_dec = DepthDecoder.DepthDecoder().double().to(device)
    weights = torch.load(args.pretrained_dispnet_dec, map_location=lambda storage, loc: storage)
    disp_net_dec.load_state_dict(weights)
    print("weights:",weights)
    disp_net_dec.eval()



    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir/file for file in f.read().splitlines()]
    else:
        test_files = sum([dataset_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])

    print('{} files to test'.format(len(test_files)))

    for file in tqdm(test_files):
        print("file",file)
        img = imread(file).astype(np.double)

        h,w,_ = img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))

        tensor_img = torch.from_numpy(img).unsqueeze(0)
        tensor_img = ((tensor_img/255 - 0.5)/0.5).to(device)
        #tensor_img = (tensor_img/255).to(device)
        #print("tensor imp:",tensor_img)

        #output_inter = disp_net_enc(tensor_img.double())
        #print("out inter:",output_inter[4].shape," len:",len(output_inter))

        #output = disp_net_dec(tensor_img.double(),output_inter)[0]
        #print("inp:",tensor_img.shape)
        econv = disp_net_enc(tensor_img.double())

        output = disp_net_dec(tensor_img.double(), econv)[0]
        print("out:",output[0])

        #print("output final: ",output.shape)

        file_path, file_ext = file.relpath(args.dataset_dir).splitext()
        file_name = '-'.join(file_path.splitall())
        #print("output:",output)
        if args.output_disp:
            disp = (255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)
            #print("finallly:", np.transpose(disp, (1, 2, 0).shape))
            imsave(output_dir/'{}_disp{}'.format(file_name, file_ext), np.transpose(disp, (1,2,0)))
        if args.output_depth:
            depth = 1/output
            depth = 255*depth
            #print("f   inallly 1:", np.transpose(depth, (1, 2, 0)).shape)
            #depth = (255*tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.double)
            #print("depth:", depth)
            #depth = depth*0
            imsave(output_dir/'{}_depth{}'.format(file_name, ".png"), np.transpose(depth, (1,2,0)))


if __name__ == '__main__':
    main()
