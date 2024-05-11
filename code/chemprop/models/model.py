from typing import List, Union

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax, degree
from .mpn_att import MPN

#from torch.nn.utils.weight_norm import weight_norm
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs, featurizer: bool = False):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param featurizer: Whether the model should act as a featurizer, i.e., outputting the
                           learned features from the last layer prior to prediction rather than
                           outputting the actual property predictions.
        """
        super(MoleculeModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.featurizer = featurizer
        self.device = args.device
        self.use_input_features = args.use_input_features

        self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        self.create_encoder(args)
        self.create_coatt(args)
        self.create_ffn(args)

        initialize_weights(self)

    def create_encoder(self, args: TrainArgs) -> None:
        """
        Creates the message passing encoder for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.encoder1 = MPN(args)
        self.encoder2 = MPN(args)

    def create_coatt(self, args: TrainArgs) -> None:
        hidden_dim = args.hidden_size * args.number_of_molecules

        self.w_j = nn.Linear(hidden_dim, hidden_dim)
        self.w_i = nn.Linear(hidden_dim, hidden_dim)

        self.prj_j = nn.Linear(hidden_dim, hidden_dim)
        self.prj_i = nn.Linear(hidden_dim, hidden_dim)
        

    def create_ffn(self, args: TrainArgs) -> None:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size  # only additional features in FFN
        else:
            first_linear_dim = args.hidden_size * args.number_of_molecules
            if args.use_input_features:
                first_linear_dim += args.features_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim*2, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim*2, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, self.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def featurize(self,
                  batch: Union[List[str], List[Chem.Mol], BatchMolGraph],
                  features_batch: List[np.ndarray] = None,
                  mol_adj_batch: List[np.ndarray] = None,
                  mol_dist_batch: List[np.ndarray] = None,
                  mol_clb_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Computes feature vectors of the input by running the model except for the last layer. (MPNEncoder + FFN[:-1])

        return: The feature vectors computed by the :class:`MoleculeModel`.
        """
        return self.ffn[:-1](self.encoder1(batch, features_batch, mol_adj_batch,
                                          mol_dist_batch, mol_clb_batch))

    def fingerprint(self,
                    batch: Union[List[str], List[Chem.Mol], BatchMolGraph],
                    features_batch: List[np.ndarray] = None,
                    mol_adj_batch: List[np.ndarray] = None,
                    mol_dist_batch: List[np.ndarray] = None,
                    mol_clb_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes the fingerprint vectors of the input molecules by passing the inputs through the MPNN and returning
        the latent representation before the FFNN. (MPNEncoder)

        return: The fingerprint vectors calculated through the MPNN.
        """
        return self.encoder1(batch, features_batch, mol_adj_batch, mol_dist_batch, mol_clb_batch)

    def forward(self,
                batch_left: Union[List[str], List[Chem.Mol], BatchMolGraph],
                batch_right:Union[List[str], List[Chem.Mol], BatchMolGraph],
                features_batch: List[np.ndarray] = None,
                features_batch_right: List[np.ndarray] = None,
                mol_adj_batch: List[np.ndarray] = None,
                mol_adj_batch_right: List[np.ndarray] = None,
                mol_dist_batch: List[np.ndarray] = None,
                mol_dist_batch_right: List[np.ndarray] = None,
                mol_clb_batch: List[np.ndarray] = None,
                mol_clb_batch_right: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :return: The output of the :class:`MoleculeModel`, which is either property predictions
                 or molecule features if :code:`self.featurizer=True`.
        """
        if self.featurizer:
            return self.featurize(batch_left, features_batch, mol_adj_batch,
                                  mol_dist_batch, mol_clb_batch)

        left_output, left_atom_output, left_batch_indices = self.encoder1(batch_left, features_batch, mol_adj_batch,
                                       mol_dist_batch, mol_clb_batch)
        right_output, right_atom_output, rightt_batch_indices = self.encoder2(batch_right, features_batch_right, mol_adj_batch_right,
                                       mol_dist_batch_right, mol_clb_batch_right)
        ##left_output shape: batch x first_linear_dim
        
        ##Use co-attention layer to compute att
        left_align= left_output.repeat_interleave(degree(rightt_batch_indices, dtype=int), dim=0)
        right_align = right_output.repeat_interleave(degree(left_batch_indices, dtype=int), dim=0)

        left_scores = (self.w_i(left_atom_output) * self.prj_i(right_align)).sum(-1)
        left_scores = softmax(left_scores, left_batch_indices)

        right_scores = (self.w_i(right_atom_output) * self.prj_i(left_align)).sum(-1)
        right_scores = softmax(right_scores, rightt_batch_indices)

        h_output = global_add_pool(left_atom_output * right_align * left_scores.unsqueeze(-1), left_batch_indices)
        t_output = global_add_pool(right_atom_output * left_align * right_scores.unsqueeze(-1), rightt_batch_indices)
        
        if self.use_input_features:
            features_batch = torch.from_numpy(
                np.stack(features_batch)).float().to(self.device)
            features_batch_right = torch.from_numpy(
                np.stack(features_batch_right)).float().to(self.device)

            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)
                features_batch_right = features_batch_right.view(1, -1)

            h_output = torch.cat([h_output, features_batch], dim=1)
            t_output = torch.cat([t_output, features_batch_right], dim=1)

        ##pertubed
        random_noise = torch.rand_like(h_output).to(h_output.device)
        h_output_pertubed = h_output + torch.sign(h_output) * nn.functional.normalize(random_noise, dim=-1) * 0.1
        t_output_pertubed = t_output + torch.sign(t_output) * nn.functional.normalize(random_noise, dim=-1) * 0.1

        pair=torch.cat([h_output_pertubed,t_output_pertubed], dim=-1)
        output = self.ffn(pair)

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            # batch size x num targets x num classes per target
            output = output.reshape((output.size(0), -1, self.num_classes))
            if not self.training:
                # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
                output = self.multiclass_softmax(output)

        return output, h_output, h_output_pertubed, left_scores, right_scores, left_output, right_output
