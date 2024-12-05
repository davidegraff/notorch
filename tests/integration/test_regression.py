"""This integration test is designed to ensure that the chemprop model can _overfit_ the training
data. A small enough dataset should be memorizable by even a moderately sized model, so this test
should generally pass."""

import warnings

from lightning import pytorch as pl
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from mol_gnn import lightning_models, nn
from mol_gnn.nn.gnn import agg
from mol_gnn.nn.message_passing import edge, embed, message, update
from mol_gnn.data import MoleculeDataset

warnings.filterwarnings("ignore", module=r"lightning.*", append=True)


@pytest.fixture(params=[agg.Sum(), agg.Gated()])
def aggr(request):
    return request.param


@pytest.fixture#(params=[nn.BondMessagePassing(), nn.AtomMessagePassing()])
def mp(aggr: agg.Aggregation):
    return edge.EdgeMessagePassing(
        embed.InputEmbedding.edge(),
        message.Identity(),
        aggr,
        update.ResidualUpdate(),
        embed.LinearOutputEmbedding(),
        torch.nn.ReLU(),
        3,
        True
    )
    return request.param


@pytest.fixture
def dataloader(random_mol_data):
    dset = MoleculeDataset(random_mol_data)
    dset.normalize_targets()

    return DataLoader(dset, 20, collate_fn=dset.collate_batch)


def test_quick(mp: nn.MessagePassing, dataloader: DataLoader):
    encoder = nn.GraphEncoder(mp, agg.Mean())
    predictor = nn.RegressionFFN()
    mpnn = lightning_models.MPNN(encoder, predictor, True)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(mpnn, dataloader, None)


def test_overfit(mp: nn.MessagePassing, dataloader: DataLoader):
    encoder = nn.GraphEncoder(mp, agg.Sum())
    predictor = nn.RegressionFFN()
    mpnn = lightning_models.MPNN(encoder, predictor, True)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        max_epochs=100,
        overfit_batches=1.00
    )
    trainer.fit(mpnn, dataloader)

    errors = []
    for batch in dataloader:
        bmg, _, _, targets, *_ = batch
        preds = mpnn(bmg)
        errors.append(preds - targets)

    errors = torch.cat(errors)
    mse = errors.square().mean().item()
    
    assert mse <= 1e-3


@pytest.mark.long
def test_lipo(mp, lipo_data):
    train_data, test_data = train_test_split(lipo_data, test_size=0.2)
    val_data, test_data = train_test_split(test_data, test_size=0.5)

    train_dataset = MoleculeDataset(train_data)
    val_dataset = MoleculeDataset(val_data)
    test_dataset = MoleculeDataset(test_data)
    scaler = train_dataset.normalize_targets()
    val_dataset.normalize_targets(scaler)

    train_loader = train_dataset.to_dataloader(
        64, num_workers=8, persistent_workers=True
    )
    val_loader = val_dataset.to_dataloader(
        64, num_workers=8, persistent_workers=True
    )

    encoder = nn.GraphEncoder(mp, agg.Sum())
    predictor = nn.RegressionFFN()
    mpnn = lightning_models.MPNN(encoder, predictor, True)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        accelerator="auto",
        devices=1,
        max_epochs=30,
    )
    trainer.fit(mpnn, train_loader, val_loader)

    Y_hats = trainer.predict(mpnn, test_dataset.to_dataloader())
    Y_hat = torch.cat(Y_hats)
    Y_hat = scaler.inverse_transform(Y_hat.cpu().numpy())

    errors = Y_hat - test_dataset.Y
    rmse = np.sqrt(np.mean(errors ** 2))
    
    assert rmse <= 0.8