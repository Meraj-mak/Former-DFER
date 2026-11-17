import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.ST_Former import GenerateModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sns
from thop import profile
from dataloader.dataset_DFEW import train_data_loader, test_data_loader

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=1, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--data_set', type=int, default=2)

args = parser.parse_args()
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H:%M]-")
log_txt_path = './log/set' + str(args.data_set) + '-log.txt'
log_curve_path = './log/set' + str(args.data_set) + '-log.png'
checkpoint_path = './checkpoint/set' + str(args.data_set) + '-model.pth'
best_checkpoint_path = './checkpoint/set' + str(args.data_set) + '-model_best.pth'
# log_txt_path = './log/' + time_str + 'set' + str(args.data_set) + '-log.txt'
# log_curve_path = './log/' + time_str + 'set' + str(args.data_set) + '-log.png'
# checkpoint_path = './checkpoint/' + time_str + 'set' + str(args.data_set) + '-model.pth'
# best_checkpoint_path = './checkpoint/' + time_str + 'set' + str(args.data_set) + '-model_best.pth'



def main():
    best_acc = 0
    recorder = RecorderMeter(args.epochs)

    war_history = []    # ←------------------------------------- add this
    uar_history = []    # ←------------------------------------- add this
    training_start_time = time.time()    # ←-------------------- add this

    print('The training time: ' + now.strftime("%m-%d %H:%M"))
    print('The training set: set ' + str(args.data_set))
    with open(log_txt_path, 'a') as f:
        f.write('The training set: set ' + str(args.data_set) + '\n')

    # create model and load pre_trained parameters
    model = GenerateModel()

    # ── TEMPORAL TRANSFORMER PARAMETER COUNT ──────────────────────────******************************** Part 1
    temporal_params = sum(p.numel() for p in model.t_former.parameters())
    temporal_trainable = sum(p.numel() for p in model.t_former.parameters() if p.requires_grad)

    print(f"Temporal Transformer - Total parameters:     {temporal_params:,}")
    print(f"Temporal Transformer - Trainable parameters: {temporal_trainable:,}")
    # ──────────────────────────────────────────────────────────────────**************************************

    # ── ENTIRE MODEL PARAMETER COUNT (what your supervisor asked for) ───────────**************************************
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nAll Model - Total parameters:     {total_params:,}")
    print(f"All Model - Trainable parameters: {total_trainable:,}")
    # ────────────────────────────────────────────────────────────**************************************

    visualize_parameters(model)    #===================================================
    compute_gflops(model)          #===================================================

    model = torch.nn.DataParallel(model).cuda()

 

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # checkpoint = torch.load(args.resume, map_location='cuda', weights_only=False)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            best_acc = best_acc.cuda()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    train_data = train_data_loader(data_set=args.data_set)
    test_data = test_data_loader(data_set=args.data_set)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)
    for i, (images, target) in enumerate(train_loader):
        print("Real input shape:", images.shape)
        break

    for epoch in range(args.start_epoch, args.epochs):
        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        print(inf)
        print('Current learning rate: ', current_learning_rate)

        # train for one epoch
        train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        #val_acc, val_los = validate(val_loader, model, criterion, args) =========================================== commit keya hai


        # evaluate on validation set ============================================
        # val_acc, val_los, all_preds, all_targets = validate(val_loader, model, criterion, args)
        val_acc, val_los, all_preds, all_targets, WAR, UAR = validate(val_loader, model, criterion, args)

        # save confusion matrix only at best epoch =====================================
        if val_acc > best_acc:
            plot_confusion_matrix(all_preds, all_targets, args.data_set)

        # WAR/UAR — record EVERY epoch regardless
        war_history.append(WAR)
        uar_history.append(UAR)
        plot_war_uar(war_history, uar_history, args.data_set)

        scheduler.step()

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best)

        # print and save log
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder.plot_curve(log_curve_path)

        print('The best accuracy: {:.3f}'.format(best_acc.item()))
        print('An epoch time: {:.1f}s'.format(epoch_time))
        with open(log_txt_path, 'a') as f:
            f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
            f.write('An epoch time: {:.1f}s' + str(epoch_time) + '\n')

    # ── TOTAL TRAINING TIME ────────────────────────────────────────────────────────────────────────────────────────────────────────────
    
    total_time_seconds = time.time() - training_start_time
    total_hours   = int(total_time_seconds // 3600)
    total_minutes = int((total_time_seconds % 3600) // 60)
    total_seconds = int(total_time_seconds % 60)

    print(f"\n{'='*50}")
    print(f"Total Training Time: {total_hours}h {total_minutes}m {total_seconds}s")
    print(f"Total Training Time: {total_time_seconds/60:.2f} minutes")
    print(f"{'='*50}")

    with open(log_txt_path, 'a') as f:
        f.write(f"\nTotal Training Time: {total_hours}h {total_minutes}m {total_seconds}s\n")
        f.write(f"Total Training Time: {total_time_seconds/60:.2f} minutes\n")
    
    # ────────────────────────────────────────────────────────────────────────────────────────────────────────────    

def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):

        images = images.cuda()
        target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # It prevents the model from making changes that are too big, which stops nan from happening. Speed Control 
        # nan (Not a Number)
        # "Never let any gradient update be stronger than 1.0"
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')
    

    # ── collect predictions for confusion matrix ────────────────────────────────────────────
    all_preds   = []
    all_targets = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # ── store predictions ────────────────────────────────────────────────────────────────────
            _, pred = output.topk(1, 1, True, True)
            all_preds.extend(pred.squeeze(1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                progress.display(i)


        # ── WAR = top1 accuracy (already computed) ──────────────
        WAR = top1.avg.item()

        # ── UAR = per-class recall averaged across 7 classes ────
        num_classes = 7
        class_correct = np.zeros(num_classes)
        class_total   = np.zeros(num_classes)

        for t, p in zip(all_targets, all_preds):
            class_total[t]   += 1
            if t == p:
                class_correct[t] += 1

        per_class_recall = class_correct / (class_total + 1e-8) * 100
        UAR = np.mean(per_class_recall)

        # ── print both ──────────────────────────────────────────────────────────────────────────────
        print(f'WAR (Accuracy): {WAR:.3f}%')
        print(f'UAR (Mean Recall): {UAR:.3f}%')

        emotion_labels = ['Happiness','Sadness','Neutral',
                          'Anger','Surprise','Disgust','Fear']
        for i, (label, recall) in enumerate(zip(emotion_labels, per_class_recall)):
            print(f'  {label:12s}: {recall:.2f}%')

        # ── save to log file ────────────────────────────────────────────────────────────────────────
        with open(log_txt_path, 'a') as f:
            f.write(f'WAR: {WAR:.3f}%\n')
            f.write(f'UAR: {UAR:.3f}%\n')
            for label, recall in zip(emotion_labels, per_class_recall):
                f.write(f'  {label}: {recall:.2f}%\n')

        # TODO: this should also be done with the ProgressMeter
        print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        with open(log_txt_path, 'a') as f:
            f.write('Current Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')
    #return top1.avg, losses.avg
    return top1.avg, losses.avg, all_preds, all_targets, WAR, UAR


def save_checkpoint(state, is_best):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('Curve was saved')
        plt.close(fig)

def visualize_parameters(model):
    s_params   = sum(p.numel() for p in model.s_former.parameters())
    t_params   = sum(p.numel() for p in model.t_former.parameters())
    fc_params  = sum(p.numel() for p in model.fc.parameters())
    all_params = sum(p.numel() for p in model.parameters())
    all_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    t_train    = sum(p.numel() for p in model.t_former.parameters() if p.requires_grad)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Former-DFER — Parameter Analysis', fontsize=16, fontweight='bold', y=0.98)

    card_data = [
        ('Temporal (t_former)\nTotal',     f'{t_params:,}',   '#534AB7'),
        ('Temporal (t_former)\nTrainable', f'{t_train:,}',    '#1D9E75'),
        ('All Model\nTotal',               f'{all_params:,}', '#534AB7'),
        ('All Model\nTrainable',           f'{all_train:,}',  '#1D9E75'),
    ]
    for i, (label, value, color) in enumerate(card_data):
        ax = fig.add_axes([0.04 + i * 0.245, 0.78, 0.22, 0.16])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        ax.add_patch(plt.Rectangle((0,0), 1, 1, color='#f5f5f5',
                                   transform=ax.transAxes, clip_on=False))
        ax.text(0.5, 0.72, label, ha='center', va='center',
                fontsize=10, color='#666666', transform=ax.transAxes)
        ax.text(0.5, 0.30, value, ha='center', va='center',
                fontsize=13, fontweight='bold', color=color, transform=ax.transAxes)

    ax_donut = fig.add_subplot(2, 2, 3)
    sizes  = [s_params, t_params, fc_params]
    colors = ['#534AB7', '#1D9E75', '#888780']
    labels = [
        f's_former\n{s_params:,}\n({s_params/all_params*100:.1f}%)',
        f't_former\n{t_params:,}\n({t_params/all_params*100:.1f}%)',
        f'fc\n{fc_params:,}\n({fc_params/all_params*100:.2f}%)'
    ]
    wedges, _ = ax_donut.pie(sizes, colors=colors, startangle=90,
                              wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
    ax_donut.legend(wedges, labels, loc='lower center',
                    bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=8, frameon=False)
    ax_donut.set_title('Parameter distribution across components',
                        fontsize=11, fontweight='bold', pad=10)

    ax_bar = fig.add_subplot(2, 2, 4)
    x     = np.arange(2)
    width = 0.35
    bars1 = ax_bar.bar(x - width/2, [t_params, all_params], width,
                        label='Total params', color='#534AB7', edgecolor='white')
    bars2 = ax_bar.bar(x + width/2, [t_train, all_train], width,
                        label='Trainable params', color='#1D9E75', alpha=0.85, edgecolor='white')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(['Temporal (t_former)', 'All model'], fontsize=10)
    ax_bar.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{v/1e6:.0f}M' if v >= 1e6 else f'{v/1e3:.0f}K'))
    ax_bar.set_title('Total vs trainable parameters', fontsize=11, fontweight='bold')
    ax_bar.legend(fontsize=9, frameon=False)
    ax_bar.spines[['top', 'right']].set_visible(False)
    ax_bar.yaxis.grid(True, color='#eeeeee'); ax_bar.set_axisbelow(True)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, h*1.01,
                    f'{h/1e6:.2f}M', ha='center', va='bottom', fontsize=8)

    # plt.tight_layout(rect=[0, 0, 1, 0.77])
    plt.subplots_adjust(left=0.05, right=0.95, top=0.75, bottom=0.1, hspace=0.4)
    plt.savefig('./log/parameter_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: ./log/parameter_analysis.png")


def compute_gflops(model):
    dummy_input = torch.randn(1, 16, 3, 112, 112)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    gflops = flops / 1e9

    print(f"\nGFLOPs: {gflops:.2f} G")
    print(f"Params (thop verify): {params/1e6:.2f} M")

    with open(log_txt_path, 'a') as f:
        f.write(f"GFLOPs: {gflops:.2f} G\n")
        f.write(f"Params (thop verify): {params/1e6:.2f} M\n")

    # ── show only GFLOPs, no redundant Params bar ──────────────
    fig, ax = plt.subplots(figsize=(4, 4))
    bar = ax.bar(['GFLOPs'], [gflops], color='#534AB7',
                 edgecolor='white', width=0.3)
    ax.text(bar[0].get_x() + bar[0].get_width()/2,
            gflops * 1.01, f'{gflops:.2f} G',
            ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.set_title('Former-DFER — Computational Complexity\n'
                 f'(Total Params: {params/1e6:.2f}M)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('GFLOPs', fontsize=11)
    ax.set_ylim(0, gflops * 1.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, color='#eeeeee')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('./log/gflops_analysis.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: ./log/gflops_analysis.png")



def plot_confusion_matrix(all_preds, all_targets, data_set):
    emotion_labels = ['Happiness', 'Sadness', 'Neutral',
                      'Anger', 'Surprise', 'Disgust', 'Fear']

    cm = confusion_matrix(all_targets, all_preds)

    # normalize to percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Confusion Matrix — Set {data_set}', fontsize=14, fontweight='bold')

    # raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_labels, yticklabels=emotion_labels,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title('Raw Counts', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=10)
    axes[0].set_ylabel('True', fontsize=10)
    axes[0].tick_params(axis='x', rotation=45)

    # normalized %
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=emotion_labels, yticklabels=emotion_labels,
                ax=axes[1], linewidths=0.5)
    axes[1].set_title('Normalized (%)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=10)
    axes[1].set_ylabel('True', fontsize=10)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    save_path = f'./log/set{data_set}-confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

def plot_war_uar(war_history, uar_history, data_set):
    epochs = list(range(1, len(war_history) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epochs, war_history, color='#534AB7', linestyle='-',
            linewidth=2, marker='o', markersize=5, label='WAR (Weighted Avg Recall)')
    ax.plot(epochs, uar_history, color='#1D9E75', linestyle='-',
            linewidth=2, marker='s', markersize=5, label='UAR (Unweighted Avg Recall)')

    # ── mark best UAR point ONLY if more than 1 epoch exists ──
    if len(uar_history) > 1:
        best_uar_epoch = int(np.argmax(uar_history)) + 1
        best_uar_val   = max(uar_history)
        offset_x = min(best_uar_epoch + 1, len(war_history))  # don't go out of range
        ax.annotate(f'Best UAR: {best_uar_val:.2f}%',
                    xy=(best_uar_epoch, best_uar_val),
                    xytext=(offset_x, best_uar_val - 5),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=9)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(f'WAR vs UAR — Set {data_set}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, color='#eeeeee')
    ax.set_axisbelow(True)

    # ── safe xlim: always add padding so low != high ──────────
    x_min = max(0.5, min(epochs) - 0.5)
    x_max = max(epochs) + 0.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    save_path = f'./log/set{data_set}-war_uar.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    main()
