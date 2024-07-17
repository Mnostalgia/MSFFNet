import os, argparse, time, datetime
import random
import warnings
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from util.MF_dataset import MF_dataset

from util.util import compute_results, visualize
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from pytorch_toolbelt import losses as L
from loss_hub.losses import DiceLoss,SoftCrossEntropyLoss
from torch.cuda.amp import autocast,GradScaler
from model import propose
#############################################################################################
parser = argparse.ArgumentParser(description='Train with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='propose')
parser.add_argument('--batch_size', '-b', type=int, default=2)
parser.add_argument('--seed', default=3407, type=int,help='seed for initializing training.')
parser.add_argument('--lr_start', '-ls', type=float, default=0.02)
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
parser.add_argument('--epoch_max', '-em', type=int, default=30)
parser.add_argument('--epoch_from', '-ef', type=int, default=0)
parser.add_argument('--num_workers', '-j', type=int, default=4)
parser.add_argument('--n_class', '-nc', type=int, default=2)
parser.add_argument('--loss_weight', '-lw', type=float, default=0.5)
parser.add_argument('--data_dir', '-dr',type=str, default='/home/sj/Desktop/dataset')
args = parser.parse_args()
#############################################################################################
#set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
scaler = GradScaler()

class JointLoss(torch.nn.Module):
    def __init__(self, dice_loss_weight=0.5, ce_loss_weight=0.5, mode='multiclass', classes=2, smooth_factor=0.1):
        super(JointLoss, self).__init__()
        self.dice_loss = DiceLoss(mode=mode, classes=classes)
        self.ce_loss = SoftCrossEntropyLoss(smooth_factor=smooth_factor)
        self.dice_loss_weight = dice_loss_weight
        self.ce_loss_weight = ce_loss_weight

    def forward(self, logits, target):
        dice_loss = self.dice_loss(logits, target)
        ce_loss = self.ce_loss(logits, target)
        joint_loss = self.dice_loss_weight * dice_loss + self.ce_loss_weight * ce_loss
        return joint_loss
    
def train(epo, model, train_loader, optimizer):
    model.train()
    for it, (images, thermal, labels, names) in enumerate(train_loader):
        images = Variable(images).cuda(args.gpu)
        thermal = Variable(thermal).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        images = torch.cat([images,thermal],dim=1)

        with autocast():
            start_t = time.time()
            optimizer.zero_grad()
            joint_loss = JointLoss(dice_loss_weight=0.5, ce_loss_weight=0.5, mode='multiclass', classes=2, smooth_factor=0.1).cuda()
            logits_S = model(images)
            loss = joint_loss(logits_S,labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        lr_this_epo = 0
        for param_group in optimizer.param_groups:
            lr_this_epo = param_group['lr']
        print('Train: %s, epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f, time %s'\
              % (args.model_name, epo, args.epoch_max, it + 1, len(train_loader), lr_this_epo,
                 len(names) / (time.time() - start_t), float(loss),
                 datetime.datetime.now().replace(microsecond=0) - start_datetime))
        if accIter['train'] % 1 == 0:
            writer.add_scalar('Train/loss', loss, accIter['train'])
        view_figure = True 
        if accIter['train'] % 500 == 0:
            if view_figure:
                input_rgb_images = vutils.make_grid(images[:, :3], nrow=8,padding=10) 
                writer.add_image('Train/input_rgb_images', input_rgb_images, accIter['train'])
                scale = max(1,255 // args.n_class)  
                groundtruth_tensor = labels.unsqueeze(1) * scale
                groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor),1) 
                groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                writer.add_image('Train/groudtruth_images', groudtruth_images, accIter['train'])
                predicted_tensor = logits_S.argmax(1).unsqueeze(1) * scale 
                predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor),1) 
                predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                writer.add_image('Train/predicted_images', predicted_images, accIter['train'])
        accIter['train'] = accIter['train'] + 1

def testing(epo, model, test_loader):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    label_list = ["unlabeled", "circle"]
    testing_results_file = os.path.join(weight_dir, 'testing_results_file.txt')
    with torch.no_grad():
        for it, (images, thermal, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            thermal = Variable(thermal).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            images = torch.cat([images, thermal], dim=1)
            logits = model(images)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten() 
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1])
            conf_total += conf
            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (
            args.model_name, epo, args.epoch_max, it + 1, len(test_loader),
            datetime.datetime.now().replace(microsecond=0) - start_datetime))
    precision, recall, IoU = compute_results(conf_total)
    writer.add_scalar('Test/average_precision', precision.mean(), epo)
    writer.add_scalar('Test/average_recall', recall.mean(), epo)
    writer.add_scalar('Test/average_IoU', IoU.mean(), epo)
    for i in range(len(precision)):
        writer.add_scalar("Test(class)/precision_class_%s" % label_list[i], precision[i], epo)
        writer.add_scalar("Test(class)/recall_culass_%s" % label_list[i], recall[i], epo)
        writer.add_scalar('Test(class)/Iou_%s' % label_list[i], IoU[i], epo)
    if epo == 0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" % (
            args.model_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write(
                "# epoch: Background, circle, average(nan_to_num). (Acc %, IoU %)\n")
    with open(testing_results_file, 'a') as f:
        f.write(str(epo) + ': ')
        for i in range(len(precision)):
            f.write('%0.4f, %0.4f, ' % (100 * recall[i], 100 * IoU[i]))
        f.write('%0.4f, %0.4f\n' % (100 * np.mean(np.nan_to_num(recall)), 100 * np.mean(np.nan_to_num(IoU))))
    print('saving testing results.')
    with open(testing_results_file, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)
    return np.mean(np.nan_to_num(IoU))


if __name__ == '__main__':
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')
    model = propose.propose(args.n_class)
    if args.gpu >= 0: model.cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)

    weight_dir = os.path.join("./runs/", args.model_name)


    writer = SummaryWriter("./runs/tensorboard_log")


    print('training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)

    train_dataset = MF_dataset(data_dir=args.data_dir, split='train')
    test_dataset = MF_dataset(data_dir=args.data_dir, split='test')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    start_datetime = datetime.datetime.now().replace(microsecond=0)
    accIter = {'train': 0, 'test': 0}



    best_iou = 0.0
    best_model_file = None
    best_epoch = None

    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' % (args.model_name, epo))
        train(epo, model, train_loader, optimizer)

        iou = testing(epo, model, test_loader)
        
        if iou > best_iou:
            best_iou = iou
            best_epoch = epo
            best_model_file = os.path.join(weight_dir, 'best.pth')
            print('Saving the best model to %s with IoU %.4f' % (best_model_file, best_iou))
            torch.save(model.state_dict(), best_model_file)
        
        
        scheduler.step()
        
    print('finish! Best model saved at epoch %d with IoU %.2f' % (best_epoch, 100*best_iou))

