from typing import List

import torch
from tqdm import tqdm
import numpy as np
from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel


def predict(model: MoleculeModel,
            data_loader: MoleculeDataLoader,
            data_right_loader: MoleculeDataLoader,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks.
    """
    model.eval()

    preds = []
    left_att = []
    right_att = []
    left_feat = []
    right_feat = []

    for batch, batch_right in tqdm(zip(data_loader, data_right_loader), disable=disable_progress_bar, total=len(data_loader), leave=False):
        # Prepare batch
        batch: MoleculeDataset
        batch_right: MoleculeDataset
        mol_batch, features_batch, mol_adj_batch, mol_dist_batch, mol_clb_batch = \
            batch.batch_graph(), batch.features(), batch.adj_features(
            ), batch.dist_features(), batch.clb_features()

        mol_batch_right, features_batch_right, mol_adj_batch_right, mol_dist_batch_right, mol_clb_batch_right = \
            batch_right.batch_graph(), batch_right.features(), batch_right.adj_features(
            ), batch_right.dist_features(), batch_right.clb_features()

        # Make predictions
        with torch.no_grad():
            batch_preds, d_feat, d_feat_pertubed, left_scores, righ_scores, left_output, right_output = model(mol_batch, mol_batch_right, features_batch, features_batch_right,
                                mol_adj_batch, mol_adj_batch_right, mol_dist_batch, mol_dist_batch_right, mol_clb_batch, mol_clb_batch_right)

        batch_preds = batch_preds.data.cpu().numpy()
        left_scores = left_scores.data.cpu().numpy()
        righ_scores = righ_scores.data.cpu().numpy()
        left_output = left_output.data.cpu().numpy()
        right_output = right_output.data.cpu().numpy()
        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()

        preds.extend(batch_preds)
        left_att.append(left_scores)
        right_att.append(righ_scores)
        left_feat.append(left_output)
        right_feat.append(right_output)

    left_att = np.concatenate(left_att, axis = 0)
    right_att = np.concatenate(right_att, axis = 0)
    left_feat = np.concatenate(left_feat, axis = 0)
    right_feat = np.concatenate(right_feat, axis = 0)

    return preds, left_att, right_att, left_feat, right_feat
