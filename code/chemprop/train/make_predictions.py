from collections import OrderedDict
import csv
from typing import List, Optional, Union
import os
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .predict import predict
from chemprop.args import PredictArgs, TrainArgs
from chemprop.data import get_data, get_data_from_smiles, MoleculeDataLoader, MoleculeDataset
from chemprop.utils import load_args, load_checkpoint, load_scalers, makedirs, timeit, update_prediction_args
from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim
from .evaluate import evaluate, evaluate_predictions

@timeit()
def make_predictions(args: PredictArgs, smiles: List[List[str]] = None) -> List[List[Optional[float]]]:
    """
    Loads data and a trained model and uses the model to make predictions on the data.

    If SMILES are provided, then makes predictions on smiles.
    Otherwise makes predictions on :code:`args.test_data`.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param smiles: List of list of SMILES to make predictions on.
    :return: A list of lists of target predictions.
    """
    print('Loading training args')
    train_args = load_args(args.checkpoint_paths[0])
    num_tasks, task_names, metrics = train_args.num_tasks, train_args.task_names, train_args.metrics
    ##task_names: ['Label']

    update_prediction_args(predict_args=args, train_args=train_args)
    args: Union[PredictArgs, TrainArgs]

    if args.atom_descriptors == 'feature':
        set_extra_atom_fdim(train_args.atom_features_size)

    if args.bond_features_path is not None:
        set_extra_bond_fdim(train_args.bond_features_size)

    print('Loading data')
    if smiles is not None:
        full_data = get_data_from_smiles(
            smiles=smiles,
            skip_invalid_smiles=False,
            features_generator=args.features_generator
        )
    else:
        test_data = get_data(path=args.test_path,
                             args=args,
                             features_path=args.features_path,
                             adjacency_path=args.adjacency_path,
                             distance_path=args.distance_path,
                             coulomb_path=args.coulomb_path,
                             smiles_columns=args.smiles_columns,
                             logger=None)
        test_data_right = get_data(path=args.test_path_right,
                             args=args,
                             features_path=args.features_path,
                             adjacency_path=args.adjacency_path,
                             distance_path=args.distance_path,
                             coulomb_path=args.coulomb_path,
                             smiles_columns=args.smiles_columns,
                             logger=None)

    test_targets = test_data.targets()
    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    print(f'Test size = {len(test_data):,}')

    # Predict with each model individually and sum predictions
    if args.dataset_type == 'multiclass':
        sum_preds = np.zeros((len(test_data), num_tasks, args.multiclass_num_classes))
    else:
        sum_preds = np.zeros((len(test_data), num_tasks))

    # Create data loader
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    test_data_loader_right = MoleculeDataLoader(
        dataset=test_data_right,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Partial results for variance robust calculation.
    if args.ensemble_variance:
        all_preds = np.zeros((len(test_data), num_tasks, len(args.checkpoint_paths)))

    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    for index, checkpoint_path in enumerate(tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths))):
        # Load model and scalers
        model = load_checkpoint(checkpoint_path, device=args.device)
        scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler = load_scalers(checkpoint_path)

        # Normalize features
        if args.features_scaling or train_args.atom_descriptor_scaling or train_args.bond_feature_scaling:
            test_data.reset_features_and_targets()
            if args.features_scaling:
                test_data.normalize_features(features_scaler)
            if train_args.atom_descriptor_scaling and args.atom_descriptors is not None:
                test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
            if train_args.bond_feature_scaling and args.bond_features_size > 0:
                test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)

        # Make predictions
        model_preds, left_att, right_att, left_feat, right_feat = predict(
            model=model,
            data_loader=test_data_loader,
            data_right_loader=test_data_loader_right,
            scaler=scaler
        )

        test_scores = evaluate_predictions(
            preds=model_preds,
            targets=test_targets,
            num_classes=args.multiclass_num_classes,
            num_tasks=num_tasks,
            metrics=metrics,
            dataset_type=args.dataset_type,
            logger=None
        )

        sum_preds += np.array(model_preds)

        for metric, scores in test_scores.items():
            avg_test_score = np.nanmean(scores)
            print(f'test {metric} = {avg_test_score:.6f}')

        if args.ensemble_variance:
            all_preds[:, :, index] = model_preds

    np.save(os.path.join(args.preds_path, 'preds.npy'), sum_preds)
    np.save(os.path.join(args.preds_path, 'left_feature.npy'), left_feat)
    np.save(os.path.join(args.preds_path, 'right_feature.npy'), right_feat)


    # Ensemble predictions
    # avg_preds = sum_preds / len(args.checkpoint_paths)
    # avg_preds = avg_preds.tolist()

    # if args.ensemble_variance:
    #     all_epi_uncs = np.var(all_preds, axis=2)
    #     all_epi_uncs = all_epi_uncs.tolist()

    # # Save predictions
    # print(f'Saving predictions to {args.preds_path}')
    # assert len(test_data) == len(avg_preds)
    # if args.ensemble_variance:
    #     assert len(test_data) == len(all_epi_uncs)
    # makedirs(args.preds_path, isfile=True)

    # # Get prediction column names
    # if args.dataset_type == 'multiclass':
    #     task_names = [f'{name}_class_{i}' for name in task_names for i in range(args.multiclass_num_classes)]
    # else:
    #     task_names = task_names

    # # Copy predictions over to full_data
    # for full_index, datapoint in enumerate(test_data):
    #     preds = avg_preds * len(task_names)
    #     if args.ensemble_variance:
    #         epi_uncs = all_epi_uncs[valid_index] if valid_index is not None else ['Invalid SMILES'] * len(task_names)

    #     # If extra columns have been dropped, add back in SMILES columns
    #     if args.drop_extra_columns:
    #         datapoint.row = OrderedDict()

    #         smiles_columns = args.smiles_columns

    #         for column, smiles in zip(smiles_columns, datapoint.smiles):
    #             datapoint.row[column] = smiles

    #     # Add predictions columns
    #     if args.ensemble_variance:
    #         for pred_name, pred, epi_unc in zip(task_names, preds, epi_uncs):
    #             datapoint.row[pred_name] = pred
    #             datapoint.row[pred_name+'_epi_unc'] = epi_unc
    #     else:
    #         for pred_name, pred in zip(task_names, preds):
    #             datapoint.row[pred_name] = pred

    # # Save
    # with open(args.preds_path, 'w') as f:
    #     writer = csv.DictWriter(f, fieldnames=test_data[0].row.keys())
    #     writer.writeheader()

    #     for datapoint in test_data:
    #         writer.writerow(datapoint.row)

    return sum_preds


def chemprop_predict() -> None:
    """Parses Chemprop predicting arguments and runs prediction using a trained Chemprop model.

    This is the entry point for the command line command :code:`chemprop_predict`.
    """
    make_predictions(args=PredictArgs().parse_args())
