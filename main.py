## Reference:
## 1. DivideMix: https://github.com/LiJunnan1992/DivideMix
## 2. CausalNL: https://github.com/a5507203/IDLN
## Our code is heavily based on the above-mentioned repositories. 

# Loading libraries
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
import random 
import argparse
import numpy as np
from models.PreResNet import *
from models.vae import *
from sklearn.mixture import GaussianMixture
import dataloader
import argparse
import os
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Default values
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--vae_lr', '--vae_learning_rate', default=0.001, type=float, help='initial vae learning rate')
parser.add_argument('--noise_mode',  default='instance')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--lambda_elbo', default=0.001, type=float, help='weight for elbo')
parser.add_argument('--lambda_ref', default=0.001, type=float, help='weight for ref')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--z_dim', default=32, type=int)
args,_ = parser.parse_known_args()
print(args)

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def unpickle(file):
    import pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

def preprocess(dataset, root_dir, r, noise_mode, replay_num=1000):
    replay_file = '%s/%.2f_%s.npy'%(root_dir,r,noise_mode)
    if os.path.exists(replay_file):
        return
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
    elif dataset=='cifar100':    
        train_dic = unpickle('%s/train'%root_dir)
        train_data = train_dic['data']
        train_label = train_dic['fine_labels']
        num_classes_ = 100
        train_data = train_data.reshape((50000, 3, 32, 32))
        train_data = train_data.transpose((0, 2, 3, 1))
    elif dataset=='fashionmnist':
        train_data = np.load('%s/train_images.npy'%root_dir)
        train_label = np.load('%s/train_labels.npy'%root_dir)
        num_classes_=10
    
    ind = np.random.randint(0, train_data.shape[0])
    data = train_data[ind]
    if dataset in ['cifar10', 'cifar100']:
        transform_train = A.ReplayCompose(
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
    elif dataset=='fashionmnist':    
            transform_train = A.ReplayCompose(
                [
                    A.ShiftScaleRotate(p=0.5),
                    A.CropAndPad(px=2, keep_size=False, always_apply=True),
                    A.RandomCrop(height=28, width=28, always_apply=True),
                    A.HorizontalFlip(),
                    A.Normalize(mean=(0.1307,), std=(0.3081)),
                    ToTensorV2(),
                ]
            )
    replay_list = []
    while True:
        if len(replay_list) < replay_num:
            img = transform_train(image=data)
            replay_list.append(img['replay'])
            if len(replay_list) == replay_num:
                print('saving replay')
                np.save(replay_file, np.array(replay_list))
                break
    id_file = '%s/%.2f_%s_id.npy'%(root_dir,r,noise_mode)
    img_id = np.arange(train_data.shape[0])
    np.random.shuffle(img_id)
    img_id+=num_classes_
    print('saving id file')
    np.save(id_file, img_id)

def factor_func(step, end_step):
    if step<end_step:
        factor = 1*step/end_step
    else:
        factor = args.lambda_ref
    return factor

# Training
def train(epoch,net,net2,optimizer,vae_model, vae_model2,optimizer_vae,labeled_trainloader,unlabeled_trainloader, train_all_loader, net_1 = True):
    net.train()
    net2.eval() #fix one network and train the other    
    vae_model.train()
    vae_model2.eval()
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    all_train_iter = iter(train_all_loader)
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, u_c_x, u_s_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2, noisy_labels, u_c_u, u_s_u, w_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, noisy_labels, u_c_u, u_s_u, w_u = unlabeled_train_iter.next()                 
        batch_size = inputs_x.size(0)
        batch_size_u = inputs_u.size(0)
        
        # Transform label to one-hot
        labels_x_onehot = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)
        noisy_labels = torch.zeros(inputs_u.size(0), args.num_class).scatter_(1, noisy_labels.view(-1,1), 1)
        w_x = w_x.view(-1,1).type(torch.FloatTensor)
        w_u = w_u.view(-1,1).type(torch.FloatTensor)

        labels_x = labels_x.cuda()
        inputs_x, inputs_x2, labels_x_onehot, u_c_x, u_s_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x_onehot.cuda(), u_c_x.cuda(), u_s_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, noisy_labels, u_c_u, u_s_u, w_u = inputs_u.cuda(), inputs_u2.cuda(), noisy_labels.cuda(), u_c_u.cuda(), u_s_u.cuda(), w_u.cuda()


        with torch.no_grad():
            # label co-guessing of unlabeled samples
            zc_mean, zc_logvar, zc, outputs_u11 = net(inputs_u)
            zc_mean, zc_logvar, zc, outputs_u12 = net(inputs_u2)
            zc_mean, zc_logvar, zc, outputs_u21 = net2(inputs_u)
            zc_mean, zc_logvar, zc, outputs_u22 = net2(inputs_u2)    
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            zc_mean, zc_logvar, zc, outputs_x = net(inputs_x)
            zc_mean, zc_logvar, zc, outputs_x2 = net(inputs_x2)         
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x_onehot + (1-w_x)*px              
            ptx = px**(1/args.T) # temparature sharpening 
                    
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
        
        mix_zc_mean, mix_zc_logvar, zc, logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]
        
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()      
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
    
        loss_dm = Lx + lamb * Lu + penalty
        try:
            vae_data, vae_targets, vae_u_c, vae_u_s, pred_clean, _ = all_train_iter.next()
        except:
            all_train_iter = iter(train_all_loader)
            vae_data, vae_targets, vae_u_c, vae_u_s, pred_clean, _ = all_train_iter.next()
        vae_loss, recons_loss, kld_loss, ce_loss = train_vae(vae_data, vae_targets,  vae_u_c, vae_u_s, pred_clean, net, vae_model)
        if epoch<40:
            factor = 0
        else:
            factor = factor_func(epoch, 140)
        loss = loss_dm + factor*ce_loss + args.lambda_elbo * vae_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        optimizer_vae.zero_grad()
        loss.backward()
        optimizer_vae.step()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f  Recons loss: %.2f  KL loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item(), Lu.item(), recons_loss.item(), kld_loss.item()))
        sys.stdout.flush()
    return loss

# Train vae
def train_vae(vae_data, vae_targets, vae_u_c, vae_u_s, pred_clean, net, vae_model):
    vae_model.train()
    data = vae_data.cuda()
    targets = vae_targets.cuda()
    u_c = vae_u_c.cuda()
    u_s = vae_u_s.cuda()
    pred_clean = pred_clean.cuda()

    x_hat, y_hat, zc_mean, zc_logvar, zs_mean, zs_logvar, p_zc_m, p_zs_m  = vae_model(data, u_c, u_s, net)
    vae_loss, recons_loss, kld_loss = my_vae_loss(x_hat, data, zc_mean, zc_logvar, zs_mean, zs_logvar, p_zc_m, p_zs_m)
    ce_loss = CE(y_hat, targets)
    ce_loss = (ce_loss*pred_clean).sum()/(pred_clean.sum()+1e-8)

    return vae_loss, recons_loss, kld_loss, ce_loss


# two component GMM model
def eval_train(model,all_loss):    
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset))    
    eval_correct = []
    with torch.no_grad():
        for _, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            zc_mean, zc_logvar, zc, outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]    
            pred = torch.max(outputs, 1)[1]
            eval_correct += (pred == targets).tolist()     
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)
    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss,eval_correct

# Testing
def test(epoch,net1,net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            zc_mean, zc_logvar, zc, outputs1 = net1(inputs)
            zc_mean, zc_logvar, zc, outputs2 = net2(inputs)   
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))  
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)


class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def create_model():
    if args.dataset == 'fashionmnist':
        model = ResNet18(num_classes=args.num_class, in_c=1, z_dim=args.z_dim)
    else:
        model = ResNet18(num_classes=args.num_class, in_c=3, z_dim=args.z_dim)
    if args.dataset=='cifar10':
        vae_model = VAE_CIFAR10(z_dim=args.z_dim, num_classes=10)
    elif args.dataset=='fashionmnist':
        vae_model = VAE_FASHIONMNIST(z_dim=args.z_dim, num_classes=10)
    elif args.dataset=='cifar100':
        vae_model = VAE_CIFAR100(z_dim=args.z_dim, num_classes=100)
    
    total_params1 = sum(p.numel() for p in model.parameters())
    total_params2 = sum(p.numel() for p in vae_model.parameters())
    print(f"Number of parameters: {total_params1+total_params2}")
    model = model.cuda()
    vae_model = vae_model.cuda()
    return model, vae_model

os.makedirs('./checkpoint', exist_ok = True)
os.makedirs('./saved/cifar10/', exist_ok= True)
os.makedirs('./saved/cifar100/', exist_ok= True)
os.makedirs('./saved/fashionmnist/', exist_ok= True)

stats_log=open('./checkpoint/%s_%.2f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
test_log=open('./checkpoint/%s_%.2f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w') 

preprocess(args.dataset,args.data_path,args.r,args.noise_mode)

if args.dataset == 'cifar10':
    warm_up = 10
    lr_decrease = 150
elif args.dataset=='fashionmnist':
    warm_up = 5
    lr_decrease = 80
    args.num_epochs=100
elif args.dataset=='cifar100':
    warm_up = 30
    lr_decrease = 150


loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=16,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.2f_%s.pt'%(args.data_path,args.r,args.noise_mode))

print('| Building net')
net1, vae_model1 = create_model()
net2, vae_model2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_vae1 = optim.Adam(vae_model1.parameters(), lr=args.vae_lr)
optimizer_vae2 = optim.Adam(vae_model2.parameters(), lr=args.vae_lr)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

all_loss = [[],[]] # save the history of losses from two networks

def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, _) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        zc_mean, zc_logvar, zc, c_logits = net(inputs)
        loss = CEloss(c_logits, labels)
        loss.backward()  
        optimizer.step() 
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.2f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
temp_ = loader.run('warmup')
img, target, _ = next(iter(temp_))


def my_vae_loss(x_hat, x, zc_mean, zc_logvar, zs_mean, zs_logvar, p_zc_m, p_zs_m):
    recons_loss = F.mse_loss(x_hat, x, reduction="mean")
    kld_loss1 = torch.mean(-0.5 * torch.sum(1 + zc_logvar - (zc_mean - p_zc_m) ** 2 - zc_logvar.exp(), dim = 1), dim = 0)
    kld_loss2 = torch.mean(-0.5 * torch.sum(1 + zs_logvar - (zs_mean - p_zs_m) ** 2 - zs_logvar.exp(), dim = 1), dim = 0)
    kld_loss =  kld_loss1+kld_loss2
    return recons_loss+kld_loss, recons_loss, kld_loss

warmup_trainloader = loader.run('warmup')
test_loader = loader.run('test')
eval_loader = loader.run('eval_train')
n_top1 = AverageMeter('Acc@1', ':6.2f')
co1_loss = AverageMeter('Acc@1', ':6.2f')
co2_loss = AverageMeter('Acc@1', ':6.2f')
vae1_loss = AverageMeter('Acc@1', ':6.2f')
vae2_loss = AverageMeter('Acc@1', ':6.2f')
test_acc = 0


start = time.time()
epoch = 0
pbar = tqdm(desc = 'Epochs', total = args.num_epochs)
while epoch < args.num_epochs:   
    lr=args.lr
    if epoch >= lr_decrease:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr
    lr=args.vae_lr
    if epoch >= lr_decrease:
        lr /= 10      
    for param_group in optimizer_vae1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_vae2.param_groups:
        param_group['lr'] = lr
    
    if epoch < warm_up:
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader)
    else:
        if epoch==warm_up:
            torch.save({
                    'epoch': epoch,
                    'net1_state_dict': net1.state_dict(),
                    'net2_state_dict': net2.state_dict(),
                    'optimizer1_state_dict': optimizer1.state_dict(),
                    'optimizer2_state_dict': optimizer2.state_dict(),
                    'vae_model1_state_dict': vae_model1.state_dict(),
                    'vae_model2_state_dict': vae_model2.state_dict(),
                    'optimizer_vae1_state_dict': optimizer_vae1.state_dict(),
                    'optimizer_vae2_state_dict': optimizer_vae2.state_dict()
                    }, './saved/%s/warmup_checkpoint_%s_%.2f'%(args.dataset, args.noise_mode, args.r)+'.tar')
        
        prob1,all_loss[0],eval_correct1=eval_train(net1,all_loss[0])   
        prob2,all_loss[1],eval_correct2=eval_train(net2,all_loss[1])            
        pred1 = (prob1 > args.p_threshold)
        pred2 = (prob2 > args.p_threshold)
        
        print('Train Net1')
        print('updating loader')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred2,prob2) # co-divide
        train_all_loader = loader.run("all",pred2,prob2)
        loss_1 = train(epoch,net1,net2,optimizer1,vae_model1,vae_model2,optimizer_vae1,labeled_trainloader, unlabeled_trainloader, train_all_loader, net_1=True) # train net1  
        
        print('\nTrain Net2')
        print('updating loader')
        labeled_trainloader, unlabeled_trainloader = loader.run('train',pred1,prob1) # co-divide
        train_all_loader = loader.run("all",pred1,prob1)
        loss_2 = train(epoch,net2,net1,optimizer2,vae_model2,vae_model1, optimizer_vae2,labeled_trainloader, unlabeled_trainloader, train_all_loader, net_1=False) # train net2     
    test(epoch,net1,net2)
    pbar.update(epoch)
    epoch += 1
    torch.save({
            'epoch': epoch,
            'net1_state_dict': net1.state_dict(),
            'net2_state_dict': net2.state_dict(),
            'optimizer1_state_dict': optimizer1.state_dict(),
            'optimizer2_state_dict': optimizer2.state_dict(),
            'vae_model1_state_dict': vae_model1.state_dict(),
            'vae_model2_state_dict': vae_model2.state_dict(),
            'optimizer_vae1_state_dict': optimizer_vae1.state_dict(),
            'optimizer_vae2_state_dict': optimizer_vae2.state_dict(),
            }, './saved/%s/checkpoint_%s_%.2f'%(args.dataset, args.noise_mode, args.r)+'.tar')
pbar.close()
end = time.time()
print(end - start)

