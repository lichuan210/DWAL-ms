# coding: utf-8
__author__ = '10900'
import mindspore as ms
import mindspore.ops as ops
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.dataset import Sampler
import mindcv
import numpy as np
import copy

from config import config
from load_dataset_svhn import *
#import load_dataset
import transform #这是自己创建的py文件

#from wideresnet_svhn import WideResNet, CNN, WNet,ResNet,CNN13 #这是自己创建的py文件

import argparse, math, time, json, os
import math
import time
import wrn_d_SVHN

import os
from data import *
from net import *
import datetime
from tqdm import tqdm



parser = argparse.ArgumentParser(description='manual to this script')

#model
parser.add_argument('--depth', type=int, default=28)
parser.add_argument('--width', type=int, default=2)


#optimization
parser.add_argument('--optim', default='adam')
parser.add_argument('--iterations', type=int, default=500000)#default=200000
parser.add_argument('--l_batch_size', type=int, default=100) #default=100
parser.add_argument('--ul_batch_size', type=int, default=100)#default=100
parser.add_argument('--test_batch_size', type=int, default=128)#default=128
parser.add_argument('--lr_decay_iter', type=int, default=400000)
parser.add_argument('--lr_decay_factor', type=float, default=0.2)
parser.add_argument('--warmup', type=int, default=200000)#default=200000
parser.add_argument('--meta_lr', type=float, default=0.001)
parser.add_argument('--lr_wnet', type=float, default=6e-5) # this parameter need to be carefully tuned for different settings

#dataset
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--n_labels', type=int, default=2400)#default=60
parser.add_argument('--n_unlabels', type=int, default=20000)#default=20000
parser.add_argument('--n_valid', type=int, default=5000)#efault=5000
parser.add_argument('--n_class', type=int, default=10)
parser.add_argument('--tot_class', type=int, default=10)
parser.add_argument('--ratio', type=float, default=0.6)
parser.add_argument("--root", "-r", default="data", type=str, help="dataset dir")

parser.add_argument("--alg", "-a", default="PI", type=str, help="ssl algorithm : [supervised, PI, MT, VAT, PL]")
parser.add_argument("--em", default=0.2, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")


args = parser.parse_args()



class MSE_Loss(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x, y, model, mask):
        y_hat = model(x)
        return (ops.mse_loss(y_hat.softmax(1), y.softmax(1).detach(), reduction='none').mean(1)*mask)
def test(model, test_loader,G):
    model.set_train(False)
    correct = 0.
    tot = 0.
    for i, data in enumerate(test_loader):
        images, labels, _ = data

        if args.dataset == 'MNIST':
            images = images.unsqueeze(1)

        images = images.float()
        labels = labels.long()

        images,tem=G(images)
        if args.dataset == 'MNIST':
         images = images.view(-1, 1, 16, 8)
        #print("C(G(image))")

        out = model(images)

        pred_label = out.max(1)[1]
        correct += (pred_label == labels).float().sum()
        tot += pred_label.size(0)
    acc = correct / tot
    return acc

def test2(model, test_loader):
        model.set_train(False)
        correct = 0.
        tot = 0.
        for i, data in enumerate(test_loader):
            images, labels,_= data
            #images=pre_process2(images)

            if args.dataset == 'MNIST':
                images = images.unsqueeze(1)

            images = images.float()
            labels = labels.long()


            # print("C(G(image))")

            out = model(images)

            pred_label = out.max(1)[1]
            correct += (pred_label == labels).float().sum()
            tot += pred_label.size(0)
        acc = correct / tot
        return acc

def pre_process(x):
    l_images = x.numpy()
    l_images = np.array(l_images, dtype=np.float)
    l_images -= np.mean(l_images)
    l_images /= np.std(l_images)
    l_images = ms.Tensor.from_numpy(l_images)
    return l_images
def pre_process2(x):
    l_images = x.numpy()
    l_images = np.array(l_images, dtype=np.float)
    m=tuple(np.mean(l_images, axis=(0, 2, 3)))
    s=tuple(np.std(l_images, axis=(0, 2, 3)))
    normalize = vision.Normalize(m,s,is_hwc=False)
    l_images = transforms.Compose([normalize])(x)
    l_images = ms.Tensor.from_numpy(l_images)
    return l_images

def pca(x):
    l_images = x.numpy()
    l_images = np.array(l_images, dtype=np.float)
    l_images -= np.mean(l_images)
    cov=np.dot(l_images.T,l_images)/l_images.shape[0]
    u,s,v=np.linalg.svd(cov)
    xrot=np.dot(l_images,u)
    xrot = ms.Tensor.from_numpy(xrot)
    return xrot

def seed_everything(seed):
    #random.seed()
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)



class RandomSampler(Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = ops.cat([ops.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

a=args.n_class

#def main():
for i in range(1):
   seed=42
   seed_everything(seed)
   args.l_batch_size = args.l_batch_size
   args.ul_batch_size = args.ul_batch_size

   condition = {}
   exp_name = ""

   print("dataset : {}".format(args.dataset))
   condition["dataset"] = args.dataset
   exp_name += str(args.dataset) + "_"

   dataset_cfg = config[args.dataset]
   transform_fn = transform.transform(*dataset_cfg["transform"])  # transform function (flip, crop, noise)

   l_train_dataset = dataset_cfg["dataset"](args.root, "l_train")
   u_train_dataset = dataset_cfg["dataset"](args.root, "u_train")
   val_dataset = dataset_cfg["dataset"](args.root, "val")
   test_dataset = dataset_cfg["dataset"](args.root, "test")

   print("labeled data : {}, unlabeled data : {}, training data : {}".format(
       len(l_train_dataset), len(u_train_dataset), len(l_train_dataset) + len(u_train_dataset)))
   print("validation data : {}, test data : {}".format(len(val_dataset), len(test_dataset)))
   condition["number_of_data"] = {
       "labeled": len(l_train_dataset), "unlabeled": len(u_train_dataset),
       "validation": len(val_dataset), "test": len(test_dataset)
   }

   shared_cfg = config["shared"]
   if args.alg != "supervised":
       # batch size = 0.5 x batch size
       l_loader = ms.dataset.GeneratorDataset(
           l_train_dataset,
           sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"]).batch(shared_cfg["batch_size"],drop_remainder=True)
       )
   else:
       l_loader = ms.dataset.GeneratorDataset(
           l_train_dataset,
           sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"]).batch(shared_cfg["batch_size"],drop_remainder=True)
       )
   print("algorithm : {}".format(args.alg))
   condition["algorithm"] = args.alg
   exp_name += str(args.alg) + "_"

   u_loader =  ms.dataset.GeneratorDataset(
       u_train_dataset,
       sampler=RandomSampler(len(u_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"]  # //2
                             )
   ).batch(shared_cfg["batch_size"],drop_remainder=True)

   val_loader = ms.dataset.GeneratorDataset(val_dataset, shuffle=False).batch(128,drop_remainder=False)
   test_loader = ms.dataset.GeneratorDataset(test_dataset,shuffle=False).batch(128,drop_remainder=False)

   print("maximum iteration : {}".format(min(len(l_loader), len(u_loader))))

   alg_cfg = config[args.alg]
   print("parameters : ", alg_cfg)
   condition["h_parameters"] = alg_cfg

   if args.em > 0:
       print("entropy minimization : {}".format(args.em))
       exp_name += "em_"
   condition["entropy_maximization"] = args.em


   G=wrn_d_SVHN.WRN2(2, 6, None)
   Gbackl=wrn_d_SVHN.GeneratorResNetyx(3)
   Gbacku=wrn_d_SVHN.GeneratorResNetyx(3)
   D=wrn_d_SVHN.adversarialnet(128)
   C=wrn_d_SVHN.WRN_C(2, 6,transform_fn)
   C2 = wrn_d_SVHN.WRN_C(2, 6, transform_fn)

   optim_g=nn.Adam(G.trainable_params(), learning_rate=3e-3,weight_decay=0.00001)
   adam_betas = (0.5, 0.999)
   optim_gl = nn.Adam(Gbackl.trainable_params(), learning_rate=3e-3)
   optim_gu = nn.Adam(Gbacku.trainable_params(), learning_rate=3e-3)
   optim_d=nn.SGD(D.trainable_params(), learning_rate=3e-3,weight_decay=0.00001)
   optimizer_c =nn.Adam(C.trainable_params(), learning_rate=3e-3,weight_decay=0.0005)
   optimizer_c2 = nn.Adam(C2.trainable_params(), learning_rate=3e-3, weight_decay=0.0001)
   scheduler = ms.train.ReduceLROnPlateau(mode='max', factor=0.9, patience=1)
   scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=100000)
   optimizer_cls = OptimWithSheduler(
       optimizer_c,
       scheduler)
   optimizer_cls2 = OptimWithSheduler(
       optimizer_c2,
       scheduler)
   xi=[]
   glossw=[]
   gloss=[]
   dlossw=[]
   dloss=[]
   priloss=[]
   lceloss=[]
   mseloss=[]
   mselossw=[]
   averageOfT=[]
   maxOfT=[]
   minOfT=[]
   SumOfPri=[]
   iteration=0
   bili=[]

   maxacc=0
   gate=0
   for i in range(1):
    print("i:{}".format(i))
    print(iteration)
    
    for l_data, u_data in zip(l_loader, u_loader):


     if iteration<0:
        l_images, l_labels,_ = l_data
        u_images, u_labels,_= u_data
        l_images, l_labels = l_images.float(), l_labels.long()
        u_images, u_labels = u_images.float(), u_labels.long()
        images = ops.cat([l_images, u_images], 0)
        labels = ops.cat([l_labels, u_labels], 0)
        C2.train()

        outc2=C2(images)
        labels[-len(u_labels):] = -1
        cls_lossc2 = ops.cross_entropy(outc2, labels, reduction='none',
                                   ignore_index=-1).mean()
        loss=cls_lossc2
        optimizer_c2.zero_grad()
        loss.backward()
        optimizer_c2.step()
        iteration=iteration+1
        if iteration==1999:
            acc=acc = test2(C2, test_loader)
            print(acc)
     else: 
        if iteration%20==0:
         xi.append(iteration)
        coef = 10.0 * math.exp(-6 * (1 - min((iteration) / args.warmup, 1)) ** 2)
        D.train()
        G.train()
        Gbackl.train()
        l_images, l_labels,_ = l_data
        u_images, u_labels,_= u_data
        l_images, l_labels = l_images.float(), l_labels.long()
        u_images, u_labels = u_images.float(), u_labels.long()
        images = ops.cat([l_images, u_images], 0)
        labels = ops.cat([l_labels, u_labels], 0)

        gimages=G(images)

        d = D(gimages)
        d=d.view(args.l_batch_size+args.ul_batch_size,1)


        w=[]

        for t in d:

            if t>0.5:
                t=t-0.5
            else:
                t=0.5-t
            t=1-2*t
            

            w.append(t)


        w=ms.Tensor(w)
        w=w.view(args.l_batch_size+args.ul_batch_size,1)


        sumOfPri = 0
        numOfPri = 0
        sumofAll=0
        for im in range(0, 100):

            sumofAll=sumofAll+w[im]
            if (labels[im]==0 or labels[im]==1 or labels[im]==2 or labels[im]==3 or labels[im]==4):
                sumOfPri = sumOfPri + w[im]
                numOfPri = numOfPri + 1
        sumOfPri = sumOfPri / numOfPri
        sumofAll = sumofAll / 100
        bl =(sumofAll-sumOfPri)/sumofAll
        gate = sumofAll


        maxim=0
        minim=1

        numacc=0
        realPriImages=[]
        numofRealPriImages=0
        for im in range(0,100):

            if(w[im]<minim):
                minim=w[im]
            if(w[im]>maxim):
                maxim=w[im]

            if(w[im]<(gate)):
                realPriImages.append(images[im].cpu())
                numofRealPriImages=numofRealPriImages+1
                if(labels[im]==0 or labels[im]==1 or labels[im]==2 or labels[im]==3 or labels[im]==4):
                    numacc=numacc+1

        numOfPri=numacc/numOfPri
        if(numofRealPriImages!=0):
         numofRealPriImages=numacc/numofRealPriImages
        realPriImages=ms.Tensor([item.cpu().detach().numpy() for item in realPriImages]).cuda()






        d_loss=0
        d_lossw=0


        for lt,wt in zip(d[:len(l_labels)],w[:len(l_labels)]):
            d_lossw = d_lossw - (wt) * ops.log(lt)
            

        for ut,wt in zip(d[len(l_labels):len(d)],w[len(l_labels):len(w)]):
            d_lossw = d_lossw - (wt) * ops.log(1-ut)
        if iteration % 20 == 0:
         dlossw.append(d_lossw.cpu().detach().numpy())
         dloss.append(d_loss)


        guimages = G(u_images)
        gimages=G(l_images)

        identity_func = nn.MSELoss()
        l_images_back=Gbackl(gimages)
        u_images_back = Gbackl(guimages)
        identity_loss=identity_func(l_images_back,l_images)+identity_func(u_images_back,u_images)
        images2 = ops.cat([l_images_back,u_images_back],0)

        cycle_func=nn.L1Loss()


        pri_loss=0
        if (numofRealPriImages != 0):
         gpriimages = G(realPriImages)




         vu = []
         for tmp_images in gpriimages:
                for tmp_images2 in guimages:
                    tmp_res = ops.norm(tmp_images.view(1, -1) - tmp_images2.view(1, -1))
                    vu.append(tmp_res)
         vu = ms.Tensor(vu)
         zu = ops.sum(vu)

         for i in range(len(vu)):
                vu[i]=ops.sigmoid(vu[i])
                pri_loss = pri_loss - ops.log(vu[i])
         pri_loss=pri_loss / (1 *1000*numacc)



        g_lossw = 0
        g_loss=0
        for lt,wt in zip(d[:len(l_labels)],w[:len(l_labels)]):
            g_lossw = g_lossw - (wt) * ops.log(lt)
        for ut,wt in zip(d[len(l_labels):len(d)],w[len(l_labels):len(w)]):
            g_lossw = g_lossw - (wt) * ops.log(1-ut)
        if iteration % 20 == 0:
         glossw.append(g_lossw.cpu().detach().numpy())
         gloss.append(g_loss)
        d=D(guimages)
        d=d.view(args.ul_batch_size,1)
        dl=D(gimages)
        dl=dl.view(args.l_batch_size,1)

        wu=[]
        for t in d:
            if t>0.5:
                t=t-0.5
            else:
                t=0.5-t
            t=1-2*t
            wu.append(t)

        wu=ops.Tensor(wu)


        wl = []
        for t in dl:
            if t > 0.5:
                t = t - 0.5
            else:
                t = 0.5 - t
            t = 1 - 2 * t
            wl.append(t)
        wl = ops.Tensor(wl)


        C2.train()
        out2 = C2(l_images_back)
        out2u=C2(u_images_back)
        cls_loss2 = ops.cross_entropy(out2, labels[0:len(l_images_back)], reduction='none',
                                    ignore_index=-1).mean()
        cls_loss2=cls_loss2 * 0.0001
        ssl_loss2=ops.softmax(out2u,axis=1)*ops.log_softmax(out2u,axis=1)
        ssl_loss2=ssl_loss2.mean()
        ssl_loss2=ssl_loss2*0.0001

        it=iteration%20
        optim_g.zero_grad()
        optim_d.zero_grad()
        optim_gu.zero_grad()
        optim_gl.zero_grad()
        if it>8:
           if iteration < 15000:
               loss = g_lossw + coef * pri_loss + 10 * identity_loss+coef * cls_loss2+coef*ssl_loss2
               loss.backward()
               optim_g.step()
               optim_gl.step()
               optim_gu.step()
           else:
               loss = g_lossw + coef * pri_loss + 10 * identity_loss + coef * cls_loss2 +coef*ssl_loss2 # 10
               loss.backward()
               optim_g.step()
               optim_gl.step()
               optim_gu.step()
        else:
           loss = d_lossw + 10 * identity_loss + coef * cls_loss2 +coef*ssl_loss2
           loss.backward()
           optim_d.step()
           optim_gl.step()

        u_labels2=copy.deepcopy(u_labels)

        C.train()
        labels[-len(u_labels):] = -1
        u_labels[:] = -1
        ssl_obj = MSE_Loss()
        out=C(images)

        u_images_back=Gbackl(guimages)
        out3=C2(u_images_back)

        out_u=out3[-len(u_labels):]
        out_u=ops.softmax(out_u,axis=1)
        out_u2=ops.sort(out_u, axis=1, descending=True)[0]
        
        
        numofWU=[]
        targetofWU=[]
        for i in range(5):
            numofWU.append(0)
            targetofWU.append(0)
        wu_weight=[]
        for i in range(5):
            wu_weight.append(0)

        for i in range(len(d)):
            m=u_labels2[i]-5
            #print(m)
            numofWU[m]=numofWU[m]+1
            targetofWU[m]=targetofWU[m]+d[i]
        for i in range(len(numofWU)):
            wu_weight[i]=targetofWU[i]/numofWU[i]

        target_margin = (
                    out_u2.max(1)[0] - out_u2[:, 1])
        target_margin=target_margin.detach()
        
        numofU=[]
        targetofU=[]
        for i in range(5):
            numofU.append(0)
            targetofU.append(0)
        c2_weight=[]
        for i in range(5):
            c2_weight.append(0)
        for i in range(len(target_margin)):
            m=u_labels2[i]-5
            numofU[m]=numofU[m]+1
            targetofU[m]=targetofU[m]+target_margin[i]
        for i in range(len(numofU)):
            c2_weight[i]=targetofU[i]/numofU[i]
        unlabeled_mask = (labels == -1).float()
        ssl_loss = ssl_obj(images, out.detach(), C, unlabeled_mask)

        L_sslw=0
        L_ssl=0
        for i in range(args.ul_batch_size):
            tmp_resw=(wu[i]+target_margin[i])*ssl_loss[i+args.ul_batch_size]
            tmp_res=ssl_loss[i+args.ul_batch_size]
            L_sslw=L_sslw+tmp_resw
            L_ssl=L_ssl+tmp_res
        if iteration % 20 == 0:
         mselossw.append(L_sslw.cpu().detach().numpy())
         mseloss.append(L_ssl)
        cls_loss = ops.cross_entropy(out, labels, reduction='none', ignore_index=-1).mean()
        outc2=C2(images)
        cls_lossc2 = ops.cross_entropy(outc2, labels, reduction='none',
                                   ignore_index=-1).mean()
        if iteration % 20 == 0:
         lceloss.append(cls_loss.cpu().detach().numpy())

        if iteration<10000:
           loss=cls_loss+coef*L_sslw+cls_lossc2
           with OptimizerManager(
                   [optimizer_cls2, optimizer_cls]):

               loss.backward()
        else:

            loss = cls_loss + coef * L_sslw+cls_lossc2# + coef*100*cls_loss2
            with OptimizerManager(
                    [optimizer_cls2, optimizer_cls]):

                loss.backward()

        if iteration%100==0:


         acc = test2(C, test_loader)


         print("acc:{}".format(acc))

         if acc>maxacc:
             maxacc = acc

        iteration += 1

if __name__ == '__main__':
    print("pre:pre_processing")
    print("lanbda max_iter=1000")