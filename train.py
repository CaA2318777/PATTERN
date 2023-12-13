import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN
from dataset import prepare_data, Dataset
from utils import *
from torchvision.utils import save_image
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.utils.multiclass import type_of_target

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_for_Denoise")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=2, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
opt = parser.parse_args()

def mse_loss(x, y, reduction='mean'):
    dif = np.square(x - y)
    if reduction == 'mean':
        return np.mean(dif)
    elif reduction == 'sum':
        return np.sum(dif)

def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    loader_train_noise = DataLoader(dataset=dataset_train, num_workers=4, batch_size=1, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # model.load_state_dict(torch.load('YOUR_NET_MODEL_PATH'), strict=True)
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    # noiseL_B=[0,55] # ingnored when opt.mode=='S'
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        if True:
            # for num, data in enumerate(loader_train_noise, 0):
            for k in range(len(dataset_train)):
                # print(len(dataset_train))
                # print(dataset_train[k][0])
                # print(dataset_train[k][0][0])
                imgn_train = torch.unsqueeze(dataset_train[k][0][0], 0)
                img_train = torch.unsqueeze(dataset_train[k][0][1], 0)
                if k == 0:
                    noise = torch.zeros(img_train.size())
                noise += imgn_train - img_train
            # noise = noise / len(dataset_train)
            noise = torch.clamp(noise, 0., 0.1)
            noise = Variable(noise.cuda())

            for k in range(len(dataset_train)):
                # training step
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                name = dataset_train[k][1]
                img_train = torch.unsqueeze(dataset_train[k][0][1], 0)
                img_train = Variable(img_train.cuda())
                # noise = imgn_train - img_train
                # imgn_train = img_train + noise
                imgn_train = torch.unsqueeze(dataset_train[k][0][0], 0)
                imgn_train = Variable(imgn_train.cuda())
                n = imgn_train - img_train
                noise = n.clone()
                # noise[n!=0] = 0.1
                # imgn_train, img_train = Variable(imgn_train.cuda()), Variable(img_train.cuda())
                # noise = Variable(noise.cuda())
                out_train_noise = model(imgn_train)
                # loss = 1 * criterion(out_train, noise) / (imgn_train.size()[0]*2)
                loss = 1 * criterion(out_train_noise, noise)
                loss.backward()
                optimizer.step()
                # results
                model.eval()
                imgn_train = torch.unsqueeze(dataset_train[k][0][0], 0)
                imgn_train = Variable(imgn_train.cuda())
                # imgn_train = torch.clamp(imgn_train - noise, 0., 1.) + noise
                out_train_noise = model(imgn_train)
                out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
                psnr_train = batch_PSNR(out_train, img_train, 1.)
                save_image(out_train_noise,f'./logs/data_out_train/{name}_{epoch+1}.png')
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                     (epoch+1, k+1, len(dataset_train), loss.item(), psnr_train))
                # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
                if step % 10 == 0:
                    # Log the scalar values
                    writer.add_scalar('loss', loss.item(), step)
                    writer.add_scalar('PSNR on training data', psnr_train, step)
                step += 1
        ## the end of each epoch
        model.eval()
        # validate
        psnr_val = 0
        psnr_val_w = 0
        psnr_val_i = 0
        loss_val = 0
        loss_val_i = 0
        loss_val_w = 0
        p = 0
        p_i = 0
        r = 0
        r_n = 0
        r_i = 0
        f1 = 0
        f1_i = 0
        num = 0
        for k in range(len(dataset_val)):
            imgn_val = torch.unsqueeze(dataset_val[k][0][0], 0)
            imgn_val = Variable(imgn_val.cuda(), volatile=True)
            # imgn_val = torch.clamp(imgn_val-noise, 0., 1.)
            # imgn_val = imgn_val + noise
            img_val = torch.unsqueeze(dataset_val[k][0][1], 0)
            imgw_val = torch.unsqueeze(dataset_val[k][0][2], 0)
            name = dataset_val[k][1]
            # print(torch.max(imgn_val), torch.min(imgn_val), torch.max(img_val), torch.min(img_val), torch.max(imgw_val), torch.min(imgw_val))
            imgn_val, img_val, imgw_val = Variable(imgn_val.cuda(), volatile=True), Variable(img_val.cuda(), volatile=True), Variable(imgw_val.cuda(), volatile=True)
            out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)

            ###########noise_mse############
            # out_val_re = imgw_val.reshape(-1)
            out_val_re = out_val.reshape(-1)
            imgn_val_re = imgn_val.reshape(-1)
            img_val_re = img_val.reshape(-1)
            out_np = out_val_re.cpu().detach().numpy()
            input_np = imgn_val_re.cpu().detach().numpy()
            gt_np = img_val_re.cpu().detach().numpy()

            loss_val += mse_loss(out_np, gt_np)             
            loss_val_i += mse_loss(input_np, gt_np)             

            ############self_create##############
            # noise_g = torch.clamp(imgn_val-imgw_val, 0., 1.).reshape(-1)
            noise_g = model(imgn_val).reshape(-1)
            noise_g_cp = noise_g.clone()
            noise_i = (imgn_val - img_val).reshape(-1)
            if torch.max(noise_i) > 0.:
                num += 1
            # print(torch.max(noise_g), torch.max(noise_i))
            # print(torch.min(noise_g), torch.min(noise_i))
            # print(torch.mean(noise_i))
            noise_g_np = noise_g.cpu().detach().numpy()
            noise_i_np = noise_i.cpu().detach().numpy()
            noise_g_np[noise_g_np>0] = int(1)
            noise_g_np[noise_g_np<=0] = int(0)
            noise_i_np[noise_i_np>0] = int(1)
            noise_i_np[noise_i_np<=0] = int(0)
            noise_g_np = noise_g_np.astype(bool).astype(int)
            noise_i_np = noise_i_np.astype(bool).astype(int)

            # print(precision_score(noise_i_np, noise_g_np), recall_score(noise_i_np, noise_g_np))
            # out_g = imgw_val.reshape(-1)
            out_g = torch.clamp(imgn_val-model(imgn_val), 0., 1.).reshape(-1)
            # out_g = torch.clamp((torch.clamp(imgn_val-torch.clamp(model(imgn_val), 0., 1.), 0., 1.) - (imgn_val - img_val)), 0., 1.).reshape(-1)
            gt = img_val.reshape(-1)
            out_g_np = out_g.cpu().detach().numpy()
            gt_np = gt.cpu().detach().numpy()
            out_g_np[out_g_np>0] = int(1)
            out_g_np[out_g_np<=0] = int(0)
            gt_np[gt_np>0] = int(1)
            gt_np[gt_np<=0] = int(0)
            out_g_np = out_g_np.astype(bool).astype(int)
            gt_np = gt_np.astype(bool).astype(int)

            interation_r = recall_score(gt_np, out_g_np)
            interation_r_n = recall_score(noise_i_np, noise_g_np)
            interation_loss_val = mse_loss(out_np, gt_np)
            interation_loss_val_i = mse_loss(input_np, gt_np)

            f = open('YOUR_PATH/record.txt', 'a')
            f.write(str(f'{epoch+1}') + '\0' + str(f'{name}') + '\0' + str(interation_r) + '\0'+ str(interation_r_n) + '\0' + str(interation_loss_val) + '\0' + str(interation_loss_val_i) + '\n')
            f.close()


            p += precision_score(gt_np, out_g_np)
            r += recall_score(noise_i_np, noise_g_np)
            r_n += recall_score(noise_i_np, noise_g_np)
            f1 += f1_score(noise_i_np, noise_g_np)

            noise_extra = (noise_g_np - noise_i_np).astype(bool)
            # print("extra_mean:", torch.mean(noise_g_cp[noise_extra]))
            noise_intra = noise_i_np.astype(bool)
            # print("intra_mean:", torch.mean(noise_g_cp[noise_intra]))


            save_image(out_val,f'./logs/data_out_val/{name}_{epoch+1}.png')
        loss_val /= len(dataset_val)
        loss_val_i /= len(dataset_val)
        # loss_val_w /= len(dataset_val)
        p /= num
        # p_i /= len(dataset_val)
        r /= num
        # r_i /= len(dataset_val)
        f1 /= num
        # f1_i /= len(dataset_val)
        if epoch == 0:
            best_loss_val = loss_val
            best_epoch = epoch+1
       
        print("\n[epoch %d] loss_val: %.8f loss_val_i: %.8f" % (epoch+1, loss_val, loss_val_i))
        print("\n[epoch %d] p: %.8f r: %.8f r_n: %.8f" % (epoch+1, p, r, r_n))
        # writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        if loss_val < best_loss_val:
            best_loss_val = loss_val
            best_epoch = epoch+1
            print('Saving model...\n')
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))
        print("\n[best epoch %d]" % (best_epoch))

  

if __name__ == "__main__":
    main()
