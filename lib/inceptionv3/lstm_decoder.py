import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from ..embedding import SPECIAL_TOKENS
from ..beam_search_helper import BeamSearchHelper


class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.1):
        super(LSTMDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=(0 if num_layers == 1 else dropout),
        )

        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, image_embeddings, caption_embeddings, caption_lengths):
        """
        This method will return the output of shape (batch_size, max_caption_length, output_size)
        max_caption_length is the maximum length of the caption in the batch
        """

        # image embedding + caption embeddings (packed so that they can be sent to LSTM at once)
        # Start with image embedding as the first time-step input
        x = torch.cat((image_embeddings.unsqueeze(1), caption_embeddings), dim=1)

        # Pack the captions embeddings to avoid training on the padded tokens
        packed = pack_padded_sequence(
            x, caption_lengths, batch_first=True, enforce_sorted=False
        )

        outputs, _ = self.lstm(packed)
        # Return the outputs with the padding
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        # mapping into token space
        outputs = self.classifier(outputs)

        return outputs

    # The embedding here is the embedding instance because we are not keeping it in this class, but we need it for the generation
    def generate(
        self,
        image_embedding,
        embedding,
        max_len,
        temperature,
        beam_width,
        top_k,
        eos_index,
    ):
        # beam search sampling helper
        helper = BeamSearchHelper(
            temperature=temperature,
            beam_width=beam_width,
            top_k=top_k,
            unk_index=embedding.stoi[SPECIAL_TOKENS["UNK"]],
            eos_index=eos_index,
            device="cpu",
        )

        # run LSTM over the inputs and predict the next token
        outputs, (h, c) = self.lstm(image_embedding)

        # As the start to the generation of the tokens we only have the image embedding, so resulting
        # logits would be of shape torch.Size([1, 84880])
        logits = self.classifier(outputs[:, -1, :])

        # repeat hidden state  and cell state `beam` times
        # The hidden state and cell state will be of dimension (num_layers, batch_size, hidden_size)
        # Repeat the hidden state for each layer to the beam_width times.
        h, c = h.repeat((1, beam_width, 1)), c.repeat((1, beam_width, 1))

        # filter `top_k` values
        logits = helper.filter_top_k(logits)

        # compute probabilities and sample k values
        sample_ind = helper.sample_k_indices(logits, k=beam_width)
        # Apply the log softmax to the filtered out values. The log softmax is applied to
        # avoid the numerical underflow problem
        sample_val = helper.filter_by_indices(logits, sample_ind).log_softmax(-1)

        sample_ind, sample_val = sample_ind.T, sample_val.T

        # define total prediction sequences
        sample_seq = sample_ind.clone().detach()

        # reusable parameters
        beam_copies = torch.tensor([beam_width] * beam_width).to(outputs.device)

        # update `has_ended` index
        # contiguous is used to ensure resulting tensor is stored in contiguous block of memory,
        # as for view to reshape it requires tensor to be stored in contiguously

        helper.has_ended = (sample_ind == eos_index).contiguous().view(-1)
        # print("helper.has_ended", helper.has_ended)

        for i in range(sample_seq.size(1), max_len):
            # predict the next time step
            # Get the embedding of the predicted tokens from the previous time-step
            inputs = embedding(sample_ind)

            # Everytime it will have 10 hidden states, as for each time step we are predicting 10 tokens
            outputs, (h, c) = self.lstm(inputs, (h, c))

            # logits would be of shape torch.Size([10, 84880]) because we want to predict tokens = beam_width
            logits = self.classifier(outputs[:, -1, :])

            (prev_seqs, prev_vals), (new_ind, new_val) = helper.process_logits(
                logits, sample_seq, sample_val
            )

            # create candidate sequences and compute their probabilities
            cand_seq = torch.cat((prev_seqs, new_ind.unsqueeze(0).T), -1)
            cand_val = prev_vals.flatten() + new_val

            # sample `beam` sequences
            filter_ind = helper.sample_k_indices(cand_val, k=beam_width)

            # update total sequences and their scores
            sample_val = cand_val[filter_ind]
            sample_seq = cand_seq[filter_ind]
            sample_ind = sample_seq[:, -1].unsqueeze(-1)

            # filter `has_ended` flags
            helper.has_ended = helper.has_ended[filter_ind]

            # check if every branch has ended
            if helper.all_ended():
                break

            # repeat hidden state `beam` times and filter by sampled indices
            h = torch.repeat_interleave(h, beam_copies, dim=1)
            c = torch.repeat_interleave(c, beam_copies, dim=1)

            h, c = h[:, filter_ind, :], c[:, filter_ind, :]

        # sample output sequence
        ind = helper.sample_k_indices(sample_val, k=2)
        output_seq = sample_seq[ind, :].squeeze()

        return output_seq
