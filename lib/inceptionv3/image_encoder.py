from torch import nn
from torchvision.models import inception_v3, Inception_V3_Weights


class ImageEncoder(nn.Module):
    def __init__(self, output_size=256, dropout=0.2):
        """Initializes ImageEncoder.

        Args:
            output_size (int): dimensions of the output embedding
            dropout (float): dropout for the encoded features
        """

        super(ImageEncoder, self).__init__()

        self.output_size = output_size

        # Use the pre-trained weights of the inception_v3 model
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)

        # Set the parameters of the Inception_V3 model to not require gradients
        for param in self.inception.parameters():
            param.requires_grad = False

        # Remove the default classifier
        self.inception.fc = nn.Identity()

        # The last layer of the inception model has 2048 features
        self.linear = nn.Linear(in_features=2048, out_features=output_size)  # self.inception.fc.out_features

        # Batch Normalization (https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
        self.batch_normalization = nn.BatchNorm1d(output_size)

        self.dropout = nn.Dropout(p=dropout)

    def train(self, mode):
        super().train(mode)

        # Keep inception always in eval mode
        self.inception.eval()

    def forward(self, images):
        """
        Encodes input images into a global image embedding.

        Args:
            images (torch.Tensor): input images of shape `[batch size, channel, width, height]`

        Returns:
            torch.Tensor: global image embedding of shape `[batch size, emb_dim]`
        """

        features = self.inception(images)

        img_embedding = self.dropout(self.batch_normalization(self.linear(features)))

        return img_embedding
