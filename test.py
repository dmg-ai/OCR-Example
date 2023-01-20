from itertools import groupby

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchaudio.functional import edit_distance
from torchaudio.models.decoder import ctc_decoder

from dataset import CapchaDataset
from model import CRNN

device = torch.device('cpu')

def test_model(
    model,
    test_ds,
    beam_search_decoder,
    number_of_test_imgs: int = 10,
    save_answer=False,
    show_answer=False,
):    
    model.eval()
    
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=number_of_test_imgs)
    x_test, y_test = next(iter(test_loader))

    y_pred = model(x_test.view(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]).to(device))
    
    _, max_index = torch.max(y_pred.permute(1,0,2), dim=2)
    beam_tokens = beam_search_decoder(y_pred.detach().cpu())

    greedy_preds, beam_preds = [], []
    for i in range(x_test.shape[0]):
        greedy_raw_prediction = list(max_index[:, i].detach().cpu().numpy())
        greedy_prediction = torch.IntTensor([c for c, _ in groupby(greedy_raw_prediction) if c != test_ds.blank_label])
        greedy_preds.append(greedy_prediction)

        beam_raw_prediction = beam_tokens[i][0].tokens
        beam_prediction = torch.IntTensor([c for c, _ in groupby(beam_raw_prediction) if c != test_ds.blank_label])
        beam_preds.append(beam_prediction)
        
    gr_cer, beam_cer = [], []
    for j in range(len(x_test)):
        actual_seq = [test_ds.classes[int(n)] for n in y_test[j].numpy() if int(n)!=test_ds.blank_label]
        greedy_seq = [test_ds.classes[int(n)] for n in greedy_preds[j].numpy() if int(n)!=test_ds.blank_label]
        beam_seq = [test_ds.classes[int(n)] for n in beam_preds[j].numpy() if int(n)!=test_ds.blank_label]

        gr_cer.append(edit_distance(actual_seq,greedy_seq)/len(actual_seq))
        beam_cer.append(edit_distance(actual_seq,beam_seq)/len(actual_seq))

        mpl.rcParams["font.size"] = 8
        plt.imshow(x_test[j], cmap="gray")
        mpl.rcParams["font.size"] = 18
        plt.gcf().text(x=0.1, y=0.3, s="Actual: " + ''.join(actual_seq))
        plt.gcf().text(x=0.1, y=0.2, s="Greedy Predicted: " + ''.join(greedy_seq))
        plt.gcf().text(x=0.1, y=0.1, s="Beam Predicted: " + ''.join(beam_seq))
        if save_answer:
            plt.savefig(f"./output/plot_{j}.png")
        if show_answer:
            plt.show()
    print('Average Greedy CER:', np.mean(gr_cer))
    print('Average Beam CER:', np.mean(beam_cer))

if __name__ == "__main__":
    test_ds = CapchaDataset((4, 5), samples=100)

    gru_hidden_size = 128
    gru_num_layers = 2
    cnn_output_height = 4
    cnn_output_width = 32

    model = model = CRNN(cnn_output_height, 
                         gru_hidden_size, 
                         gru_num_layers, 
                         test_ds.num_classes).to(device)
    model_state_dict = torch.load("./checkpoints/norm_checkpoint_5.pt", map_location=device)
    model.load_state_dict(model_state_dict)

    beam_search_decoder = ctc_decoder(
        lexicon=None,
        tokens=test_ds.classes,
        nbest=1,
        beam_size=5,
        blank_token=test_ds.blank_token,
        sil_token=test_ds.blank_token,
    )

    test_model(
        model,
        test_ds,
        beam_search_decoder,
        number_of_test_imgs=10,
        save_answer=False,
        show_answer=True,
    )