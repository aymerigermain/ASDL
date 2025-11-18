# coding: utf-8

"""
Dans ce TP, nous implémentons un réseau convolutif U-Net pour résoudre un
problème de ségmentation sémantique utilisant la base de données Pascal VOC

Ce TP permet de se confronter à une tâche plus complexe que la classification du premier TP
en particulier en ce qui concerne la mise en oeuvre d'un réseau tel qu'UNet.
"""


# Standard imports
import os

# External imports
import click
import torch.nn as nn
import torch
import deepcs
import deepcs.display
import deepcs.training
from deepcs.testing import test
from deepcs.fileutils import generate_unique_logpath
from deepcs.metrics import BatchAccuracy, BatchF1
from torch.utils.tensorboard import SummaryWriter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Local imports
import data
import models
import losses


def wrap_loss(loss):
    def wrapped_loss(predictions, targets):
        targets = targets.long()
        return loss(predictions, targets)

    return wrapped_loss


@click.command(context_settings={"show_default": True})
@click.option(
    "--batch_size", type=int, default=16, help="Le nombre d'échantillons par minibatch"
)
@click.option(
    "--normalize",
    type=bool,
    default=True,
    help="Doit on normaliser les données d'entrées?",
)
@click.option(
    "--num_epochs", type=int, default=50, help="Le nombre d'époques d'entrainement"
)
@click.option(
    "--learning_rate",
    type=float,
    default=0.001,
    help="Le taux d'apprentissage de base",
)
@click.option(
    "--modelname",
    type=click.Choice(["DeepLabV3", "UNet"]),
    default="UNet",
    help="Le modèle à entrainer",
)
@click.option(
    "--run_name", type=str, default="run", help="Un préfixe pour identifier ce run"
)
@click.option(
    "--seed", type=int, default=None, help="La graine du générateur aléatoire"
)
@click.option(
    "--crop_size", type=int, default=256, help="La taille des vignettes à considérer"
)
def train(
    batch_size,
    normalize,
    num_epochs,
    modelname,
    learning_rate,
    run_name,
    seed,
    crop_size,
):
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chargement des données
    dataset_dir = os.path.join("/mounts/datasets/datasets/Pascal-VOC2012")
    (
        train_loader,
        valid_loader,
        (train_mean, train_std),
        train_transforms,
    ) = data.get_trainvalid_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        normalize=normalize,
        crop_size=crop_size,
        num_workers=num_workers,
    )
    """     test_loader = data.get_test_dataloader(
        dataset_dir=dataset_dir,
        mean=train_mean,
        std=train_std,
        batch_size=batch_size,
        num_workers=num_workers,
    ) """

    # Pour récupérer un minibatch :
    X, y = next(iter(train_loader))
    # Quelles sont les dimensions des tenseurs X et y ?
    print(X.shape, y.shape)

    cin = X.shape[1]
    input_size = X.shape[1:]
    num_classes = 21

    # Affichage de quelques échantillons
    data.plot_samples(dataset_dir)

    # Modèle
    if modelname == "DeepLabV3":
        model = models.DeepLabV3(cin, num_classes)
    elif modelname == "UNet":
        model = models.UNet(cin, num_classes)

    model = model.to(device)

    # Fonction de perte et optimiseur
    # loss = wrap_loss(nn.CrossEntropyLoss(ignore_index=255))
    loss = wrap_loss(losses.FocalLoss(ignore_index=255))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Callbacks, logging, etc..
    logdir = generate_unique_logpath("./logs", run_name)
    os.makedirs(logdir)

    train_fmetrics = {
        "CE": deepcs.metrics.GenericBatchMetric(loss),
        "accuracy": BatchAccuracy(),
        "F1": BatchF1(),
    }
    test_fmetrics = {
        "CE": deepcs.metrics.GenericBatchMetric(loss),
        "accuracy": BatchAccuracy(),
        "F1": BatchF1(),
    }

    # Affiche des informations concernant l'entrainement
    summary_text = (
        "Summary of the model architecture\n"
        + "=================================\n"
        + f"{deepcs.display.torch_summarize(model, (batch_size, ) + input_size)}\n\n"
        + "Train transforms\n"
        + "================\n"
        + f"{train_transforms}\n\n"
    )
    print(summary_text)

    # On sauvegarde les informations concernant ce run
    with open(os.path.join(logdir, "summary.txt"), "w") as fh:
        fh.write(summary_text)

    # Callbacks
    tensorboard_writer = SummaryWriter(log_dir=logdir)
    tensorboard_writer.add_text(
        "Experiment summary", deepcs.display.htmlize(summary_text)
    )

    model_checkpoint = deepcs.training.ModelCheckpoint(
        model, os.path.join(logdir, "best_model.pt")
    )

    # Boucle d'entrainement
    for e in range(num_epochs):
        print(f"[Epoch {e}/{num_epochs}]")

        # On entraine sur une epoch
        deepcs.training.train(
            model,
            train_loader,
            loss,
            optimizer,
            device,
            train_fmetrics,
            tensorboard_writer=tensorboard_writer,
        )

        # Puis on calcule les métriques d'entrainement, de validation et de test
        # Pour le pli d'entrainement
        train_metrics = test(
            model, train_loader, device, test_fmetrics, dynamic_display=False
        )
        for bname, bm in test_fmetrics.items():
            bm.tensorboard_write(tensorboard_writer, f"metrics/train_{bname}", e)

        # Pour le pli de validation
        valid_metrics = test(
            model, valid_loader, device, test_fmetrics, dynamic_display=False
        )
        for bname, bm in test_fmetrics.items():
            bm.tensorboard_write(tensorboard_writer, f"metrics/valid_{bname}", e)
        updated = model_checkpoint.update(valid_metrics["CE"])

        # On affiche les métriques dans la console
        metrics_msg = "- Train : \n  "
        metrics_msg += "\n  ".join(
            f" {m_name}: {m_value}" for (m_name, m_value) in train_metrics.items()
        )
        metrics_msg += "\n"
        metrics_msg += "- Valid : \n  "
        metrics_msg += "\n  ".join(
            f" {m_name}: {m_value}"
            + ("[>> BETTER <<]" if updated and m_name == "CE" else "")
            for (m_name, m_value) in valid_metrics.items()
        )
        print(metrics_msg)

        # On logge sur le tensorboard quelques exemples de prédiction sur le pli de validation
        model.eval()
        with torch.no_grad():
            X_val, y_val = next(iter(valid_loader))

            X_val = X_val.to(device)
            y_prob_pred = model(X_val)  # B, num_classes, H, W
            y_pred = y_prob_pred.argmax(dim=1).cpu()

            nrows = max(10, batch_size)
            ncols = 3
            fig, axes = plt.subplots(
                figsize=(6, 18), facecolor="w", nrows=nrows, ncols=ncols
            )
            cmap = data.color_map()

            for i in range(nrows):
                input_i, _ = valid_loader.dataset.dataset[i]
                input_i = np.array(input_i)  # PIL -> nd array
                pred_i = y_pred[i]
                gt_i = y_val[i]

                ax = axes[i, 0]
                ax.imshow(input_i)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                ax = axes[i, 1]
                colored_gt = data.colorize(cmap, gt_i.squeeze())
                ax.imshow(colored_gt)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                ax = axes[i, 2]
                colored_pred = data.colorize(cmap, pred_i.squeeze())
                ax.imshow(colored_pred)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.savefig("sample_prediction.png", bbox_inches="tight")
            tensorboard_writer.add_figure(
                "Sample prediction on the validation fold", fig, global_step=e
            )

        scheduler.step(valid_metrics["CE"])
        lr = scheduler.get_last_lr()
        print(
            f"Current learning rate : {lr}, num bad epochs = {scheduler.num_bad_epochs}"
        )

        print("\n\n")


if __name__ == "__main__":
    train()
