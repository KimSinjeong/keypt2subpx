import os

import torch
import torch.optim as optim
from tqdm import trange
from tensorboardX import SummaryWriter
from pathlib import Path

from model import AttnTuner
from utils import calculate_loss, create_log_dir, last_checkpoint
from test import valid
from logger import Logger
from dataset import SparsePatchDataset
from settings import get_config, get_logger, print_usage

CUDA_LAUNCH_BLOCKING=2, 3
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def train_step(step, optimizer, model, data, writer, mask=None):
    corr, _, gt_E, _, _, K1s, K2s, _, _, patch1, patch2, descriptor1, descriptor2 = data[:13]
    scorepatch1, scorepatch2 = (data[13].cuda(), data[14].cuda()) if len(data) > 13 else (None, None)
    meanft = ((descriptor1 + descriptor2) / 2.).cuda()
    idx1 = 2.5 * K1s[:,:2,:2].unsqueeze(1).cuda().inverse() @ model(patch1.cuda(), scorepatch1, meanft).unsqueeze(-1)
    idx1 = idx1.squeeze(-1)
    idx2 = 2.5 * K2s[:,:2,:2].unsqueeze(1).cuda().inverse() @ model(patch2.cuda(), scorepatch2, meanft).unsqueeze(-1)
    idx2 = idx2.squeeze(-1)
    inp = corr[...,:4].cuda() + torch.cat([idx1, idx2], 2) # B x N x 4
    loss = calculate_loss(inp, gt_E.cuda(), K1s.cuda(), K2s.cuda(), opt.train_thr)
        
    writer.add_scalar('train/loss', loss.item(), step+1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return [loss.item()], mask

def train(model, train_loader, valid_loader, opt, writer, logger):
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    checkpoint_path = last_checkpoint(opt.model_path)
    opt.resume = opt.resume and bool(checkpoint_path) and os.path.isfile(checkpoint_path)
    if opt.resume:
        logger.info('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        best_acc = checkpoint['best_acc']
        start_step = checkpoint['step']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        logger_train = Logger(opt.log_path / 'log_train.txt', title='k2s', resume=True)
        logger_valid = Logger(opt.log_path / 'log_valid.txt', title='k2s', resume=True)

    else:
        start_step = 0

        logger_train = Logger(opt.log_path / 'log_train.txt', title='k2s')
        logger_train.set_names(['Learning Rate'] + ['Sampson Loss'])
        logger_valid = Logger(opt.log_path / 'log_valid.txt', title='k2s')
        logger_valid.set_names(['AUC5'] + ['AUC10', 'AUC20'])

        aucs5, aucs10, aucs20, _, _, _, loss = valid(valid_loader, model, opt, writer, start_step)
        logger_valid.append([aucs5, aucs10, aucs20])

        best_acc = -loss
        logger.info(f"Saving initial model with va_res = {best_acc:6.3f}")
        if start_step == 0:
            torch.save({
                'step': start_step,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, os.path.join(opt.model_path, f'checkpoint_{start_step}.pth'))

    train_loader_iter = iter(train_loader)

    tbar = trange(start_step, opt.train_iter)

    mask = None

    for step in tbar:
        try:
            train_data = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            train_data = next(train_loader_iter)

        # run training
        cur_lr = optimizer.param_groups[0]['lr']

        loss_vals, mask = train_step(step, optimizer, model, train_data, writer, mask)

        tbar.set_description('Doing: {}/{}, LR: {}, Sampson_loss: {}'\
        .format(step+1, opt.train_iter, cur_lr, loss_vals[0]))

        if step % 100 == 0:
            logger_train.append([cur_lr] + loss_vals)

        # Check if we want to write validation
        b_save = ((step+1) % opt.save_intv) == 0
        b_validate = ((step+1) % opt.val_intv) == 0

        if b_validate:
            aucs5, aucs10, aucs20, _, _, _, loss = valid(valid_loader, model, opt, writer, step)
            logger_valid.append([aucs5, aucs10, aucs20])

            va_res = -loss
            if va_res > best_acc:
                logger.info(f"Saving best model with va_res = {best_acc:6.3f}")
                best_acc = va_res
                torch.save({
                'step': step + 1,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                }, os.path.join(opt.model_path, 'model_best.pth'))

            model.train()

        if b_save:
            torch.save({
            'step': step + 1,
            'model': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            }, os.path.join(opt.model_path, f'checkpoint_{step+1}.pth'))

if __name__ == "__main__":
    # parse command line arguments
    # If we have unparsed arguments, print usage and exit
    opt, unparsed = get_config()
    if len(unparsed) > 0:
        print_usage()
        exit(1)
    logger = get_logger(opt)

    # construct folder that should contain pre-calculated correspondences
    train_data = opt.datasets.split(',') #support multiple training datasets used jointly
    logger.info('Using datasets: ' + ', '.join(train_data))

    valid_data = train_data
    train_data = ['data/' + ds + '/' + 'train_' + opt.detector + '/' for ds in train_data]
    valid_data = ['data/' + ds + '/' + 'val_' + opt.detector + '/' for ds in valid_data]

    with_score = opt.detector in ['spnn', 'splg', 'aliked']

    trainset = SparsePatchDataset(train_data, opt.nfeatures, opt.fmat, not opt.sideinfo, not with_score, train=True)
    valset = SparsePatchDataset(valid_data, opt.nfeatures, opt.fmat, not opt.sideinfo, not with_score)
    train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=8, batch_size=opt.batchsize)
    valid_loader = torch.utils.data.DataLoader(valset, shuffle=False, num_workers=8, batch_size=1)

    logger.info(f"Image pairs: {len(trainset)}")

    # create or load model
    lentable = {
        'spnn': 256,
        'splg': 256,
        'aliked': 128,
        'dedode': 256,
        'xfeat': 64
    }
    model = AttnTuner(lentable[opt.detector], with_score)
    if len(opt.model) > 0:
        model.load_state_dict(torch.load(opt.model)['model'])
    model = model.cuda()
    model.train()

    # ----------------------------------------
    logger.info(f"Starting experiment {opt.experiment}")
    output_dir = Path(__file__).parent / 'results' / opt.experiment
    output_dir.mkdir(exist_ok=True, parents=True)
    create_log_dir(output_dir, opt)
    writer = SummaryWriter(log_dir=str(output_dir))

    if opt.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    train(model, train_loader, valid_loader, opt, writer, logger)
