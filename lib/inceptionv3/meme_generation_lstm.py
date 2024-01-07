import torch
from torch import nn
from .image_encoder import ImageEncoder
from .lstm_decoder import LSTMDecoder
from ..embedding import embedding, SPECIAL_TOKENS


class MemeGenerationLSTM(nn.Module):
    """LSTM-based image captioning model.

    Encodes input images into a embeddings of size `emb_dim`
    and passes them as the first token to the caption generation decoder.
    """

    def __init__(self, embedding, hidden_size, num_layers, encoder_dropout, decoder_dropout):
        super(MemeGenerationLSTM, self).__init__()

        # We need to keep them same for LSTM because the first
        # input to the LSTM is the image embedding and rest is caption
        # embedding
        self.embedding = embedding
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout

        # Initialize the Encoder which is the Image Encoder (ResNet50 model)
        self.encoder = ImageEncoder(
            output_size=self.embedding.embedding_dim,  # Embedding dimension of the Image which is 50 defined in variable EMBEDDING_DIMENSIONS
            dropout=self.encoder_dropout,
        )

        self.decoder = LSTMDecoder(
            input_size=self.embedding.embedding_dim,  # Embedding dimension of the Image which is 50 defined in variable EMBEDDING_DIMENSIONS
            hidden_size=self.hidden_size,
            output_size=self.embedding.num_embeddings,  # Number of tokens in the vocabulary including the special tokens
            num_layers=self.num_layers,
            dropout=self.decoder_dropout,
        )

        self.params = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "encoder_dropout": encoder_dropout,
            "decoder_dropout": decoder_dropout,
        }

    def forward(self, images, captions, caption_lengths):
        # Retrieve Image Embeddings for the processed batch
        image_embeddings = self.encoder(images)
        # Retrieve Caption Embeddings for the processed batch
        caption_embeddings = self.embedding(captions)
        # Get the outputs from the decoder
        output = self.decoder(image_embeddings, caption_embeddings, caption_lengths)

        return output

    def generate(self, image, max_len, temperature, beam_width, top_k):
        """Generates caption for an image.

        Args:
            image (torch.Tensor): input image of shape `[1, width, height]`
            caption (torch.Tensor, optional): beginning tokens of the caption of shape `[1, seq_len]`
            max_len (int): maximum length of the caption
            temperature (float): temperature for softmax over logits
            beam_width (int): number of maintained branches at each step
            top_k (int): number of the most probable tokens to consider during sampling

        Returns:
            torch.Tensor: generated caption tokens of shape `[1, min(output_len, max_len)]`
        """

        # get image embedding
        image_embedding = self.encoder(image).unsqueeze(1)

        sampled_ids = self.decoder.generate(
            image_embedding,
            embedding=self.embedding,
            max_len=max_len,
            temperature=temperature,
            beam_width=beam_width,
            top_k=top_k,
            eos_index=self.embedding.stoi[SPECIAL_TOKENS["EOS"]],
        )

        return sampled_ids

    def save(self, ckpt_path):
        """Saves the model's state and hyperparameters."""
        torch.save({"model": self.state_dict(), "params": self.params}, ckpt_path)

    @staticmethod
    def from_pretrained(ckpt_path):
        """Loads and builds the model from the checkpoint file."""
        ckpt = torch.load(ckpt_path, map_location="cpu")
        params = ckpt["params"]

        model = MemeGenerationLSTM(
            embedding=embedding,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            encoder_dropout=params["encoder_dropout"],
            decoder_dropout=params["decoder_dropout"],
        )
        model.load_state_dict(ckpt["model"])

        return model
