from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import pandas as pd
from PIL import Image
import os
import pickle
import torch
import tools
from torchnet.meter import AUCMeter

import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings("ignore")

            
def unpickle(file):
    import pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform_weak, transform_strong,mode, noise_file='', pred=[], probability=[], log=''): 
        
        self.r = r 
        self.transform = transform_weak
        self.transform_strong = transform_strong
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        self.replay_list = []
        self.replay_num = 1000
        self.replay_file = '%s/%.2f_%s.npy'%(root_dir,r,noise_mode)
        self.id_file = '%s/%.2f_%s_id.npy'%(root_dir,r,noise_mode)
        if os.path.exists(self.replay_file):
            print('loading replay')
            self.replay_list = np.load(self.replay_file, allow_pickle=True)
            print('loading id file')
            self.u_c_list = np.load(self.id_file, allow_pickle=True)
            
     
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
                num_classes_ = 10
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']          
                num_classes_ = 100          
            elif dataset=='fashionmnist':
                self.test_data = np.load('%s/test_images.npy'%root_dir)
                self.test_label = np.load('%s/test_labels.npy'%root_dir)   
                num_classes_ = 10
        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
                num_classes_ = 10
                train_data = train_data.reshape((50000, 3, 32, 32))
                train_data = train_data.transpose((0, 2, 3, 1))
                feature_size = 32*32*3
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
                num_classes_ = 100
                train_data = train_data.reshape((50000, 3, 32, 32))
                train_data = train_data.transpose((0, 2, 3, 1))
                feature_size = 32*32*3
            elif dataset=='fashionmnist':
                train_data = np.load('%s/train_images.npy'%root_dir)
                train_label = np.load('%s/train_labels.npy'%root_dir)
                num_classes_ = 10
                feature_size = 28*28

            if noise_mode in ['worse_label', 'aggre_label', 'random_label1', 'random_label2', 'random_label3']:
                noise_label = torch.load('./CIFAR-N/CIFAR-10_human.pt') 
                worst_label = noise_label['worse_label'] 
                aggre_label = noise_label['aggre_label'] 
                random_label1 = noise_label['random_label1'] 
                random_label2 = noise_label['random_label2'] 
                random_label3 = noise_label['random_label3']
                print('loading %s'%(noise_mode))
                noise_label = noise_label[noise_mode]
            elif noise_mode == 'noisy_label':
                noise_label = torch.load('./CIFAR-N/CIFAR-100_human.pt') 
                print('loading %s'%(noise_mode))
                noise_label = noise_label[noise_mode]
            else:
                if os.path.exists(noise_file):
                    print('loading %s'%noise_file)
                    noise_label = torch.load(noise_file)
                else:
                    data_ = torch.from_numpy(train_data).float().cuda()
                    targets_ = torch.IntTensor(train_label).cuda()
                    dataset = zip(data_, targets_)
                    if noise_mode == 'instance':
                        train_label = torch.FloatTensor(train_label).cuda()
                        noise_label = tools.get_instance_noisy_label(self.r, dataset, train_label, num_classes = num_classes_, feature_size = feature_size, norm_std=0.1, seed=123)
                    elif noise_mode == 'sym':
                        noise_label = []
                        idx = list(range(train_data.shape[0]))
                        random.shuffle(idx)
                        num_noise = int(self.r*train_data.shape[0])            
                        noise_idx = idx[:num_noise]
                        for i in range(train_data.shape[0]):
                            if i in noise_idx:
                                noiselabel = random.randint(0,num_classes_-1)
                                noise_label.append(noiselabel)
                            else:    
                                noise_label.append(train_label[i])   
                        noise_label = np.array(noise_label)
                    elif noise_mode == 'pair':
                        train_label = np.array(train_label)
                        train_label = train_label.reshape((-1,1))
                        noise_label = tools.noisify_pairflip(train_label, self.r, 123, num_classes_)
                        noise_label = noise_label[:, 0]
                    print("save noisy labels to %s ..."%noise_file)     
                    torch.save(noise_label, noise_file)   

            
            if  self.mode == 'warmup':
                self.train_data = train_data
                self.noise_label = noise_label
            elif self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
                self.pred_idx = pred.nonzero()[0]
                # updating id
                y_cluster_id_map={}
                self.cluster_id_map={}
                for i in range(train_data.shape[0]):
                    if i in self.pred_idx:
                        if noise_label[i] not in y_cluster_id_map.keys():
                            y_cluster_id_map[noise_label[i]]=self.u_c_list[i]
                        self.cluster_id_map[self.u_c_list[i]]=noise_label[i]
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    # updating id
                    y_cluster_id_map={}
                    self.cluster_id_map={}
                    self.u_c_list
                    for i in range(train_data.shape[0]):
                        if i in pred_idx:
                            if noise_label[i] not in y_cluster_id_map.keys():
                                y_cluster_id_map[noise_label[i]]=self.u_c_list[i]
                            self.cluster_id_map[self.u_c_list[i]]=noise_label[i]
                    
                    clean = (np.array(noise_label)==np.array(train_label))                                                   
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value()
                    print('Numer of labeled samples:%d   AUC:%.3f'%(pred.sum(),auc))

                    log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    log.flush()      
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                 
                    self.probability = [probability[i] for i in pred_idx]                               
                
                self.train_data = train_data[pred_idx]
                self.u_c_list = self.u_c_list[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            u_s = random.randint(0, self.replay_num-1)
            img1 = self.transform(image=img)
            img2 = self.transform_strong(image=img)
            img1=img1['image']
            img2=img2['image']
            u_c = self.u_c_list[index]
            if u_c in self.cluster_id_map.keys():
                u_c=self.cluster_id_map[u_c]
            return img1, img2, target, u_c, u_s, prob
        elif self.mode=='unlabeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            u_s = random.randint(0, self.replay_num-1)
            img1 = self.transform(image=img)
            img2 = self.transform_strong(image=img)
            img1=img1['image']
            img2=img2['image']
            u_c = self.u_c_list[index]
            return img1, img2, target, u_c, u_s, prob
        elif self.mode=='warmup':
            img, target = self.train_data[index], self.noise_label[index]
            img = self.transform(image=img)
            img=img['image']
            return img, target, index        
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            if len(self.replay_list) < self.replay_num:
                img = self.transform_strong(image=img)
                self.replay_list.append(img['replay'])
                if len(self.replay_list) == self.replay_num:
                    print('saving replay')
                    np.save(self.replay_file, np.array(self.replay_list))
                u_s=len(self.replay_list)-1
            else:
                u_s = random.randint(0, self.replay_num-1)
                img = A.ReplayCompose.replay(self.replay_list[u_s], image=img)
            u_c = self.u_c_list[index]
            if u_c in self.cluster_id_map.keys():
                u_c=self.cluster_id_map[u_c]
            img=img['image']

            if index in self.pred_idx:
                pred_clean = 1
            else:
                pred_clean=0
            return img, target, u_c, u_s, pred_clean, index        
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = self.transform(image=img)
            img=img['image']
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        if self.dataset in ['cifar10', 'cifar100']:
            self.transform_train = A.ReplayCompose(
                [
                    A.ShiftScaleRotate(p=0.5),
                    A.CropAndPad(px=4, keep_size=False, always_apply=True),
                    A.RandomCrop(height=32, width=32, always_apply=True),
                    A.HorizontalFlip(),
                    A.RandomBrightnessContrast(p=0.5),
                    A.ColorJitter(0.8, 0.8, 0.8, 0.2,p=0.8),
                    A.ToGray(p=0.2),
                    A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                    ToTensorV2(),
                ]
            )
            self.transform_train_norm = A.Compose([
                    A.CropAndPad(px=4, keep_size=False, always_apply=True),
                    A.RandomCrop(height=32, width=32, always_apply=True),
                    A.HorizontalFlip(),
                    A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                    ToTensorV2(),
                ]) 
            self.transform_test = A.Compose(
                [
                    A.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                    ToTensorV2(),
                ]
            )
        elif self.dataset=='fashionmnist':    
            self.transform_train = A.ReplayCompose(
                [
                    A.ShiftScaleRotate(p=0.5),
                    A.CropAndPad(px=2, keep_size=False, always_apply=True),
                    A.RandomCrop(height=28, width=28, always_apply=True),
                    A.HorizontalFlip(),
                    A.Normalize(mean=(0.1307,), std=(0.3081)),
                    ToTensorV2(),
                ]
            )
            self.transform_train_norm = A.Compose([
                    A.CropAndPad(px=2, keep_size=False, always_apply=True),
                    A.RandomCrop(height=28, width=28, always_apply=True),
                    A.HorizontalFlip(),
                    A.Normalize(mean=(0.1307,), std=(0.3081)),
                    ToTensorV2(),
                ])
            self.transform_test = A.Compose(
                [
                    A.Normalize(mean=(0.1307,), std=(0.3081)),
                    ToTensorV2(),
                ]
            )
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform_weak=self.transform_train_norm, transform_strong=self.transform_train, mode="warmup",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader

        elif mode=='all':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform_weak=self.transform_train_norm, transform_strong=self.transform_train, mode="all",noise_file=self.noise_file, pred=pred, probability=prob)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform_weak=self.transform_train_norm, transform_strong=self.transform_train, mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,log=self.log)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform_weak=self.transform_train_norm, transform_strong=self.transform_train, mode="unlabeled", noise_file=self.noise_file, pred=pred, probability=prob)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform_weak=self.transform_test, transform_strong=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform_weak=self.transform_test, transform_strong=self.transform_test, mode='warmup', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        
