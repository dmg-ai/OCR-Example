import sys
from itertools import groupby

import torch
import torch.nn as nn
from colorama import Fore
from torchaudio.models.decoder import ctc_decoder
from tqdm import tqdm

from dataset import CapchaDataset
from model import CRNN

device = torch.device("cpu")
epochs = 5

gru_hidden_size = 128
gru_num_layers = 2
cnn_output_height = 4
cnn_output_width = 32

model_save_path = "./checkpoints"


def train_one_epoch(model, criterion, optimizer, data_loader) -> None:
    model.train()
    train_correct = 0
    train_total = 0
    for x_train, y_train in tqdm(
            data_loader,
            position=0,
            leave=True,
            file=sys.stdout,
            bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET),
    ):
        batch_size = x_train.shape[0]  # x_train.shape == torch.Size([64, 28, 140])
        x_train = x_train.view(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])

        optimizer.zero_grad()
        y_pred = model(x_train.to(device))        
        y_pred = y_pred.permute(1, 0, 2)  # y_pred.shape == torch.Size([64, 32, 11])

        input_lengths = torch.IntTensor(batch_size).fill_(cnn_output_width)
        target_lengths = torch.IntTensor([len(t) for t in y_train])
        loss = criterion(y_pred, y_train, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        
        _, max_index = torch.max(y_pred, dim=2)
        for i in range(batch_size):
            raw_prediction = list(max_index[:, i].detach().cpu().numpy())
            prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != train_ds.blank_label])
            if is_correct(prediction, y_train[i], train_ds.blank_label):
                train_correct += 1
            train_total += 1
    print("TRAINING. Correct: ", train_correct, "/", train_total, "=", train_correct / train_total)


def is_correct(prediction, y_true, blank):
    prediction = prediction.to(torch.int32)
    prediction = prediction[prediction != blank]
    y_true = y_true.to(torch.int32)
    y_true = y_true[y_true != blank]
    return len(prediction) == len(y_true) and torch.all(prediction.eq(y_true))


def evaluate(model, val_loader, beam_search_decoder) -> float:
    model.eval()
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        for x_val, y_val in tqdm(
                val_loader,
                position=0,
                leave=True,
                file=sys.stdout,
                bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET),
        ):
            batch_size = x_val.shape[0]
            x_val = x_val.view(x_val.shape[0], 1, x_val.shape[1], x_val.shape[2])
            y_pred = model(x_val.to(device))

            beam_tokens = beam_search_decoder(y_pred.detach().cpu())
            for i in range(batch_size):
                raw_prediction = beam_tokens[i][0].tokens
                prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != train_ds.blank_label])
                if is_correct(prediction, y_val[i], train_ds.blank_label):
                    val_correct += 1
                val_total += 1
        acc = val_correct / val_total
        print("TESTING. Correct: ", val_correct, "/", val_total, "=", acc)
    return acc


if __name__ == "__main__":
    train_ds = CapchaDataset((3, 5), samples=100000)
    val_ds = CapchaDataset((3, 5), samples=100)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64)

    model = CRNN(cnn_output_height, 
                 gru_hidden_size, 
                 gru_num_layers, 
                 train_ds.num_classes).to(device)

    criterion = nn.CTCLoss(blank=train_ds.blank_label, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    beam_search_decoder = ctc_decoder(
        lexicon=None,
        tokens=train_ds.classes,
        nbest=1,
        beam_size=5,
        blank_token=train_ds.blank_token,
        sil_token=train_ds.blank_token,
    )

    current_acc = 0
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}/{epochs}")
        train_one_epoch(model, criterion, optimizer, train_loader)
        acc = evaluate(model, val_loader, beam_search_decoder)
        if acc > current_acc:
            model_out_name = model_save_path + f"/checkpoint_{epoch}.pt"
            torch.save(model.state_dict(), model_out_name)