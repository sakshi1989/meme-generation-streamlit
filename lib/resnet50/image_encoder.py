from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class ImageEncoder(nn.Module):
    def __init__(self, output_size=256, dropout=0.2):
        """Initializes ImageEncoder.

        Args:
            output_size (int): dimensions of the output embedding
            dropout (float): dropout for the encoded features
        """

        super(ImageEncoder, self).__init__()

        self.output_size = output_size

        # Use the pre-trained weights of the ResNet50 model
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Set the parameters of the ResNet50 model to not require gradients
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(
            in_features=self.resnet.fc.out_features, out_features=output_size
        )

        # Batch Normalization (https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
        self.batch_normalization = nn.BatchNorm1d(output_size)

        self.dropout = nn.Dropout(p=dropout)

    def train(self, mode):
        super().train(mode)

        # Keep resnet always in eval mode
        self.resnet.eval()

    def forward(self, images):
        """
        Encodes input images into a global image embedding.

        Args:
            images (torch.Tensor): input images of shape `[batch size, channel, width, height]`

        Returns:
            torch.Tensor: global image embedding of shape `[batch size, emb_dim]`
        """

        features = self.resnet(images)

        img_embedding = self.dropout(self.batch_normalization(self.linear(features)))

        return img_embedding
