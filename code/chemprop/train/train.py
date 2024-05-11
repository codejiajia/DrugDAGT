import logging
from typing import Callable

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from .nt_xent import NT_Xent
from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.models import MoleculeModel
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR


def train(model: MoleculeModel,
          data_loader: MoleculeDataLoader,
          data_right_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch. (epoch = iterations x batch)

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param loss_func: Loss function.
    :param optimizer: An optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for training the model.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for recording output.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """

    debug = logger.debug if logger is not None else print

    model.train()
    loss_sum = iter_count = 0

    for batch, batch_right in tqdm(zip(data_loader, data_right_loader), total=len(data_loader), leave=False):
        # Prepare batch
        batch: MoleculeDataset
        batch_right: MoleculeDataset

        mol_batch, features_batch, target_batch, mol_adj_batch, mol_dist_batch, mol_clb_batch = \
            batch.batch_graph(), batch.features(), batch.targets(), batch.adj_features(), \
            batch.dist_features(), batch.clb_features()

        mol_batch_right, features_batch_right, target_batch_right, mol_adj_batch_right, mol_dist_batch_right, mol_clb_batch_right = \
            batch_right.batch_graph(), batch_right.features(), batch_right.targets(), batch_right.adj_features(), \
            batch_right.dist_features(), batch_right.clb_features()

#         print(target_batch[10])
#         print(mol_clb_batch[10])
        mask = torch.Tensor([[x is not None for x in tb]
                            for tb in target_batch])

        targets = torch.Tensor(
            [[0 if x is None else x for x in tb] for tb in target_batch])
        targets = torch.zeros(targets.size(0),args.multiclass_num_classes).scatter_(1, targets.long(), 1)

        # Run model
        model.zero_grad()  # hignlights
        preds, d_feat, d_feat_pertubed, left_scores, right_scores, left_output, right_output = model(mol_batch, mol_batch_right, features_batch, features_batch_right, mol_adj_batch, mol_adj_batch_right,
                      mol_dist_batch, mol_dist_batch_right, mol_clb_batch, mol_clb_batch_right)  # hignlights

        # Move tensors to correct device
        mask = mask.to(preds.device)
        targets = targets.to(preds.device)
        class_weights = torch.ones(targets.shape, device=preds.device)

        if args.dataset_type == 'multiclass':
            #targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets).unsqueeze(
                1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
        else:
            loss = loss_func(preds, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()
        criterion = NT_Xent(len(batch), 0.1, 1)
        cl_loss = 0.05 * criterion(d_feat, d_feat_pertubed)
        loss += cl_loss
        loss_sum += loss.item()
        iter_count += 1

        loss.backward()  # hignlights
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()  # hignlights update weights

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(batch)

        # Log and/or add to tensorboard
        # n_iter: Iterations is the number of batches needed to complete one epoch. (n_iter == num_batch ;dataset = batch_size x n_iter)
        # print the result every 10 iterations
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            # Computes the norm of the parameters of a model.
            pnorm = compute_pnorm(model)
            # Computes the norm of the gradients of a model.
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum = iter_count = 0

            lrs_str = ', '.join(
                f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(
                f'Loss = {loss_avg:.4e}, Par_Norm = {pnorm:.4f}, Grad_Norm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter
