"""This integration test is designed to ensure that the chemprop model can _overfit_ the training
data. A small enough dataset should be memorizable by even a moderately sized model, so this test
should generally pass."""

import warnings

from lightning import pytorch as pl
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from mol_gnn import featurizers, models, nn
from mol_gnn.nn.message_passing import edge, embed, message, update, agg
from mol_gnn.data import MoleculeDatapoint, MoleculeDataset

# warnings.simplefilter("ignore", category=UserWarning, append=True)
warnings.filterwarnings("ignore", module=r"lightning.*", append=True)

@pytest.fixture
def data(smis: list):
    Y = np.random.randn(len(smis), 1)

    return [MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, Y)]


@pytest.fixture(params=[agg.Sum(), agg.GatedAttention()])
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
def dataloader(data: list[MoleculeDatapoint]):
    featurizer = featurizers.BaseMoleculeMolGraphFeaturizer()
    dset = MoleculeDataset(data, featurizer)
    dset.normalize_targets()

    return DataLoader(dset, 20, collate_fn=dset.collate_batch)


def test_quick(mp: nn.MessagePassing, dataloader: DataLoader):
    encoder = nn.GraphEncoder(mp, agg.Mean())
    predictor = nn.RegressionFFN()
    mpnn = models.MPNN(encoder, predictor, True)

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
    encoder = nn.GraphEncoder(mp, agg.Mean())
    predictor = nn.RegressionFFN()
    mpnn = models.MPNN(encoder, predictor, True)

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
    
    assert mse <= 0.05