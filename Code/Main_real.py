import SharedEncoder
import DepthDecoder
import PoseNetwork_real
import WarpImage
import LossCalc
import torch
from torch import optim
import numpy as np
import torch.nn.functional as F
import glob
import cv2


sub_dirs=[]
sub_dir_idx = 0
pose_type =1
load_img0,load_img1,load_img2, camera_intrinsics=None,None,None,None
train_folders= []
val_folders= []
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def LossBackpropagation(optimizer,loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def EstimatePose(model,depth_enc):
    dof, trans, rotation = model( depth_enc[4][0:len(depth_enc[4])-2],depth_enc[4][1:len(depth_enc[4])-1], depth_enc[4][2:len(depth_enc[4])])
    return dof,trans,rotation

def PhotometricLoss(all_images, depth_maps,camera_intrinsics,dof_prev,dof_next,ssim_fn):
    global device
    cam_intrinsics = torch.from_numpy(camera_intrinsics).to(device)
    i=0

    loss=0
    ssim=0
    l1=0
    for d_map in depth_maps:
        if i==0:
            s1 = all_images[0:len(all_images)-2]
            s2 = all_images[2:len(all_images)]
            target_scaled =  all_images[1:len(all_images)-1]

        else:
            orginal_scaled = F.interpolate(all_images, (d_map.size(2), d_map.size(3)), mode='area')



            s1 = orginal_scaled[0:len(orginal_scaled) - 2]
            s2 = orginal_scaled[2:len(orginal_scaled)]
            target_scaled = orginal_scaled[1:len(orginal_scaled) - 1]
        downscale = 128/target_scaled.shape[3]

        intrinsics_scaled = torch.cat((cam_intrinsics[:, 0:2] / downscale, cam_intrinsics[:, 2:]), dim=1)

        i=i+1



        d_map_t = d_map.view(d_map.shape[0],d_map.shape[2],d_map.shape[3])

        gen_img_s1, vec = WarpImage.inverse_warp(s1, d_map_t[1:len(d_map_t)-1], dof_prev,intrinsics_scaled, rotation_mode='euler',padding_mode='zeros')

        gen_img_s2, vec = WarpImage.inverse_warp(s2, d_map_t[1:len(d_map_t)-1], dof_next,intrinsics_scaled, rotation_mode='euler',padding_mode='zeros')


        loss_s1, ssim_s1, l1_s1 = LossCalc.calculateLoss(target_scaled, gen_img_s1, ssim_fn)
        loss_s2, ssim_s2, l1_s2 = LossCalc.calculateLoss(target_scaled, gen_img_s2, ssim_fn)

        loss += ((loss_s1 + loss_s2))
        ssim += ((ssim_s1+ssim_s2)/2)
        l1 += ((l1_s1+l1_s2)/2)
    smooth_loss = LossCalc.smooth_loss(depth_maps)
    loss =(loss/8) + (smooth_loss)
    return loss,ssim,l1,smooth_loss.item()
#LOAD ALL CAERA INTRINSICS REQUIRED FOR THE PARTICULAR FOLDER
def LoadCameraIntrinsics(folder):
    x =[]
    matrix = np.ones(shape=(3,3))
    file = open("dump/"+folder+"/cam.txt", "r")
    i = 0
    for f in file:
        x = f.split(" ")
        matrix[i,0]=  float(x[0])
        matrix[i,1] = float(x[1])
        matrix[i,2] = float(x[2])
        i=i+1

    file.close()
    return matrix

#LOAD ALL IMAGES FROM THE PARTICULAR FOLDER
def LoadData(folder):
    cam_incs=[]
    cam_inc = LoadCameraIntrinsics(folder)
    images = [((cv2.imread(file)/255-0.5)/0.5) for file in sorted(glob.glob('dump/'+folder+'/*jpg'))]
    for i in range(len(images)):
        cam_incs.append(cam_inc)
    return images, np.asarray(cam_incs)

#LOAD ALL TRAIN DATA FOLDERS FROM train.txt FILE
def InitTrainData():
    base = "dump"
    global train_folders
    file = open("dump/train.txt", "r")
    for f in file:
        train_folders.append(f[:-1])
    #print(train_folders)
    file.close()
#LOAD ALL VAL DATA FOLDERS FROM val.txt FILE
def InitValData():
    base = "dump"
    global val_folders
    file = open("dump/val.txt", "r")
    for f in file:
        val_folders.append(f[:-1])
    file.close()
#INITIALIZE ALL THE NECESSARY MODELS
def InitModels():
    global device
    # Depth shared encoder
    model_enc = SharedEncoder.SharedEncoderMain()
    # Depth decoder
    model_depth_dec = DepthDecoder.DepthDecoder().double()
    #Pose model
    pose_model = PoseNetwork_real.PoseDecoder().double()

    optimizer = optim.Adam([
        {'params': model_depth_dec.parameters(), 'lr': 1e-4, 'momentum': 0.9},
        {'params': model_enc.parameters(), 'lr': 1e-4, 'momentum': 0.9},
        {'params': pose_model.parameters(), 'lr': 1e-4, 'momentum': 0.9}], lr=1e-2)

    return model_enc.to(device), model_depth_dec.to(device), pose_model.to(device),optimizer


def trianNetworks(img,pose_model,model_enc,model_depth_dec,train_val,no_of_epoch,train_batch,val_batch,iter,camera_intrinsics,log,ssim_fn,optimizer):
    global device
    global pose_type
    b = len(img)
    img_width,img_height,_ =  img[0].shape

    t = np.zeros(shape=(b, 3, img_height, img_width),
                 dtype=np.uint8)
    t = np.transpose(np.asarray(img), (0, 3, 2, 1))

    t = torch.from_numpy(t).type(torch.DoubleTensor).to(device)



    # Depth encoder
    model_enc.train()
    econv_t = model_enc(t)

    # Depth decoder
    model_depth_dec.train()
    disp_t = model_depth_dec(t, econv_t)
    depth_t = [1 / disp for disp in disp_t]

    # Pose estimation
    dof, trans, rot = EstimatePose(pose_model, econv_t)

    # Reshaping DOF
    dof_prev = dof[:, 0, :]
    dof_next = dof[:, 1, :]

    # Image warping and Photometric loss
    loss,ssim,l1,smooth = PhotometricLoss(t, depth_t, camera_intrinsics, dof_prev, dof_next,ssim_fn)
    store_loss= loss.item()

    if train_val == 0:
        if iter%5==0:
            print(" Epoch:", no_of_epoch, " Batch:", iter, "Loss: ", store_loss," Ssim:",ssim," l1:",l1," smooth:",smooth)
        log.write(" Epoch:"+ str(no_of_epoch)+ " Batch:"+ str(iter)+ " Loss: "+ str(store_loss)+" Ssim:"+str(ssim)+" l1:"+str(l1)+" Smooth:"+str(smooth)+"\n")
        LossBackpropagation(optimizer, loss)
        loss=0
    else:
        log.write("Validation Epoch:" + str(no_of_epoch) + " Batch:" + str(iter) + " Loss: " + str(store_loss) + " Ssim:" + str(ssim) + " l1:" + str(l1) + " Smooth:" + str(smooth) + "\n")
        if iter%5==0:
            print("Validation Batch:", iter, " Loss:", store_loss," Ssim:",ssim," l1:",l1," smooth:",smooth)
    return store_loss


def saveModels(model,name):
    torch.save(model.state_dict(),"Model_real/"+name)

def createLogFile():
    return open("log_real.txt","w+")

def closeLogFile(f):
    f.close()

def main():
    global train_folders
    global val_folders

    TOTAL_EPOCH =15
    TOTAL_TRAIN = 56
    TOTAL_VAL = 5
    VAL_TRAIN = 2

    #Create log file
    log = createLogFile()

    #Init Train and Validation data
    InitTrainData()
    InitValData()

    #Initialize SSIM loss function
    ssim_fn = LossCalc.initSSim()

    #Initialize all the models necessary
    model_enc,model_depth_dec,pose_model,optimizer  = InitModels()

    no_of_epoch = 0
    #START EACH EPOCH
    while no_of_epoch < TOTAL_EPOCH:
        #FOR EACH EPOCH DO TRAINING AND VALIDATION
        for train_val in range(VAL_TRAIN): #0- train , 1 - validation
            train_batch = 0
            val_batch = 0
            id = 0
            epoch_loss =0
            #FOR TRAIN/VAL DO TILL NO OF BATCHES MENTIONED
            while train_batch < TOTAL_TRAIN and val_batch < TOTAL_VAL:
                if train_val ==0:
                    load_img,camera_intrinsics = LoadData(train_folders[train_batch])
                else:
                    load_img, camera_intrinsics = LoadData(val_folders[val_batch])
                if train_val==0:
                    train_batch+=1
                else:
                    val_batch+=1
                i = 0
                #EACH BATCH DIVIDE INTO MINIBATCH OF SIZE 8 AND THEN PROCEED
                while i+8<len(load_img):
                    epoch_loss+=trianNetworks(load_img[i:(i+8)], pose_model, model_enc, model_depth_dec, train_val, no_of_epoch, train_batch,
                              val_batch,id,camera_intrinsics[i:(i+6)],log,ssim_fn,optimizer)
                    i=i+7
                    id+=1
                if train_val==0:
                    train_batch+=1
                else:
                    val_batch+=1

            if train_val==0:
                print("Training Epoch:",no_of_epoch, " Total loss:",epoch_loss/id)
            else:
                print("Validation Epoch:", no_of_epoch, " Total loss:", epoch_loss / id)

        saveModels(model_depth_dec, "DepthDecoder/" + "model_{0:03d}.pwf".format(no_of_epoch))
        saveModels(model_enc, "DepthEncoder/" + "model_{0:03d}.pwf".format(no_of_epoch))
        saveModels(pose_model, "PoseDecoder/" + "model_{0:03d}.pwf".format(no_of_epoch))
        no_of_epoch=no_of_epoch+1
    closeLogFile(log)
main()
