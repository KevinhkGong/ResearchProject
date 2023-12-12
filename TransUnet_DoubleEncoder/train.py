import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
# import wandb
from evaluate import evaluate, evaluate_final
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from unet.vit_seg_modeling import VisionTransformer as Vit_seg
from unet.vit_seg_modeling import CONFIGS as CONFIGS_Vit_seg

dir_img = Path('./data/imgs_train/')
dir_mask = Path('./data/masks_train/')
dir_checkpoint = Path('./checkpoints/')
dir_img_test = Path('./data/imgs_test/')
dir_mask_test = Path('./data/masks_test/')
dir_pseudo_train = Path('./data/pseudo_train_optic/')
dir_pseudo_test = Path('./data/pseudo_test_optic/')
num_epoch = []
train_loss = []
evaluation_dice = []
epoch_losses = []
# Initialization
bce_loss = BCEWithLogitsLoss()
ce_loss = CrossEntropyLoss()

def graph_losses(epoch, train, evaluation, lr, aucRoc, aucPr, best_epoch, best_eval_dice):
    fig, ax = plt.subplots(1, 2, figsize=(20, 12))

    ax[0].plot(epoch, train, label='train loss', marker='o', linestyle='-')
    title = "Train Loss with lr=" + str(lr)
    ax[0].set_title(title)
    ax[0].set_ylabel('train loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend()

    ax[1].plot(epoch, evaluation, label='evaluation dice', marker='o', linestyle='--')
    title = "Evaluation Dice with lr=" + str(lr)
    ax[1].set_title(title)
    ax[1].set_ylabel('evaluation dice')
    ax[1].set_xlabel('epoch')
    ax[1].legend()

    # Adding a custom sentence at the bottom of the entire figure
    sentence = ("Auc_Roc score: " + str(format(aucRoc, '.6f')) + ", Auc_Pr score:" +
                str(format(aucPr, '.6f')) + ", Best_epoch:" + str(best_epoch)
                + ", Best eval dice score: " + str(format(best_eval_dice, '.6f')))
    fig.text(0.5, 0.01, sentence, ha='center')

    plt.tight_layout()
    plt.show()


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask,dir_pseudo_train, img_scale)
        dataset_test = CarvanaDataset(dir_img_test, dir_mask_test,dir_pseudo_test, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask,dir_pseudo_train, img_scale)
        dataset_test = BasicDataset(dir_img_test, dir_mask_test,dir_pseudo_test, img_scale)

    # 2. Split into train / validation partitions
    n_val = len(dataset_test)
    n_train = len(dataset)
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_set = dataset
    val_set = dataset_test
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(
    #     dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #          val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    # )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    pos_weight = torch.tensor([5.0]).to('cuda:1')
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    global_step = 0
    times = 1
    best_epoch = 0
    best_evaluation_dice = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):

        model.train()
        epoch_loss = 0
        t_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                # assert images.shape[1] == model.n_channels, \
                #     f'Network has been defined with {model.n_channels} input channels, ' \
                #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        # # add bce LOSS
                        # loss = bce_loss(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        # loss = criterion(masks_pred, true_masks)
                        # Use CrossEntropyLoss for multi-class classification
                        loss = ce_loss(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                t_loss = loss.item()

        # Evaluation round
        division_step = (n_train // (5 * batch_size))
        if division_step > 0:

        #    if global_step % division_step == 0:
        #     histograms = {}
        #     for tag, value in model.named_parameters():
        #         tag = tag.replace('/', '.')
        #         if not (torch.isinf(value) | torch.isnan(value)).any():
        #             histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
        #         if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
        #             histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            val_score = evaluate(model, val_loader, device, amp)
            scheduler.step(val_score)

            # graphing addition
            # record the last loss
            num_epoch.append(times)
            evaluation_dice.append(val_score.item())
            train_loss.append(t_loss)
            if val_score.item() > best_evaluation_dice:
                best_epoch = times
                best_evaluation_dice = val_score.item()

            times += 1

            logging.info('Validation Dice score: {}'.format(val_score))
            # try:
            #     experiment.log({
            #         'learning rate': optimizer.param_groups[0]['lr'],
            #         'validation Dice': val_score,
            #         'images': wandb.Image(images[0].cpu()),
            #         'masks': {
            #             'true': wandb.Image(true_masks[0].float().cpu()),
            #             'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
            #         },
            #         'step': global_step,
            #         'epoch': epoch,
            #         **histograms
            #     })
            # except:
            #     pass

        if save_checkpoint:
            if best_epoch == epoch:
                folder_path = str(dir_checkpoint)
                files = os.listdir(folder_path)
                for file in files:
                    file_path = os.path.join(folder_path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')
            if epoch == epochs:
                state_dict = model.state_dict()
                state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')


    ### ###################load best checkpoint
    # load TransUNet
    name = 'R50-ViT-B_16'
    config_vit = CONFIGS_Vit_seg[name]

    if name.find('R50') != -1:
        config_vit.patches.grid = (int(512 / 16), int(512 / 16))
    model_path = 'checkpoints/checkpoint_epoch' + str(best_epoch) + '.pth'
    # eval_net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    eval_net = Vit_seg(config_vit, img_size=512, n_classes=config_vit.n_classes).cuda()
    eval_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    eval_net.to(device=eval_device)
    eval_state_dict = torch.load(model_path, map_location=device)
    mask_values = eval_state_dict.pop('mask_values', [0, 1])
    eval_net.load_state_dict(eval_state_dict)
    ### ###################Evaluate
    aucRoc, aucPr = evaluate_final(eval_net, val_loader, device, amp)

    graph_losses(num_epoch, train_loss, evaluation_dice, learning_rate, aucRoc, aucPr, best_epoch,
                 best_evaluation_dice)



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # model = model.to(memory_format=torch.channels_last)
    # changing for TransUNet
    name = 'R50-ViT-B_16'
    config_vit = CONFIGS_Vit_seg[name]
    if name.find('R50') != -1:
        config_vit.patches.grid = (int(512 / 16), int(512 / 16))
    model = Vit_seg(config_vit, img_size=512, n_classes=config_vit.n_classes).cuda()
    model.load_from(weights=np.load(config_vit.pretrained_path))



    logging.info(f'Network:\n'
                 f'\t{model.n_classes} output channels (classes)\n')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
