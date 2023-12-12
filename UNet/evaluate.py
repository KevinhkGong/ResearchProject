import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    auc_scores = []

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']



            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image).squeeze()

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                """
                mask_true_np = mask_true.squeeze().detach().cpu().numpy()
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'

                unique_classes_binary = np.unique(mask_true_np)
                mask_pred_probs = F.sigmoid(mask_pred).squeeze().detach().cpu().numpy()

                if len(unique_classes_binary) > 1:
                    auc_scores.append(roc_auc_score(mask_true_np, mask_pred_probs))

                mask_pred = (mask_pred_probs > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                """
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                
                """
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_softmax = F.softmax(mask_pred, dim=1)
                for class_idx in range(1, net.n_classes):  # skipping background
                    mask_true_class = mask_true[:, class_idx].squeeze().detach().cpu().numpy()

                    unique_classes = np.unique(mask_true_class)
                    if len(unique_classes) > 1:
                        mask_pred_class = mask_pred_softmax[:, class_idx].squeeze().detach().cpu().numpy()
                        auc_scores.append(roc_auc_score(mask_true_class, mask_pred_class))

                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                """

    net.train()
    avg_auc_score = sum(auc_scores) / len(auc_scores) if auc_scores else 0
    return dice_score / max(num_val_batches, 1)


@torch.inference_mode()
def evaluate_final(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    all_preds = []
    all_trues = []
    aucroc, aucpr = -1, -1


    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                all_preds.append(mask_pred.view(-1).cpu().numpy())
                all_trues.append(mask_true.view(-1).cpu().numpy())
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                for c in range(net.n_classes):
                    all_preds.append(mask_pred[:, c, :, :].view(-1).cpu().numpy())
                    all_trues.append(mask_true[:, c, :, :].view(-1).cpu().numpy())

    if net.n_classes > 1:
        auc_scores = []
        auc_pr = []
        for c in range(net.n_classes):
            auc_c = roc_auc_score(all_trues[c], all_preds[c])
            precision, recall, thresholds = precision_recall_curve(all_trues[c], all_preds[c])
            pr = auc(recall, precision)
            auc_pr.append(pr)
            auc_scores.append(auc_c)
        auc_score = np.mean(auc_scores)
        auc_pc_score = np.mean(auc_pr)
        print("AUC Score (inside):", auc_score)
        print("AUC_PR Score (inside):", auc_pc_score)
        aucroc = auc_score
        aucpr = auc_pc_score
    else:
        # For single channel binary classification, compute AUC directly
        all_preds_flat = np.concatenate(all_preds)
        all_trues_flat = np.concatenate(all_trues)
        auc_score = roc_auc_score(all_trues_flat, all_preds_flat)
        precision, recall, thresholds = precision_recall_curve(all_preds_flat, all_trues_flat)
        pr = auc(recall, precision)
        print("AUC_ROC Score:", auc_score)
        print("AUC_PR Score:", pr)
        aucroc = auc_score
        aucpr = pr


    net.train()



    return aucroc, aucpr
