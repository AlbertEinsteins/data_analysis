# conv classifier

import torch
from torch import nn
from torchvision.models import resnet34
import torch.utils.data

class NNClassifier(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.classifier = nn.Sequential(
            resnet34(pretrained=True),
            nn.LazyLinear(n_classes)
        )

    def forward(self, x):
        return self.classifier(x)

    def predict(self, x, batch_size=16):
        """
        Need to load pretrained model first
        :param x:
        :return: numpy pred
        """
        # transfer to torch
        N = x.shape[0]
        x = x.reshape(x.shape[0], 28, 28)
        x = torch.as_tensor(x, dtype=torch.float)
        x = (x / 255.).unsqueeze(1)

        self.eval()

        test_dataloader = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=False, num_workers=0)
        device = next(self.parameters()).device
        predict_y = None
        with torch.no_grad():
            for val_data in test_dataloader:
                val_images = val_data.repeat(1, 3, 1, 1)

                outputs = self(val_images.to(device))  # eval model only have last output layer
                # loss = loss_function(outputs, test_labels)
                if predict_y is None:
                    predict_y = torch.max(outputs, dim=1)[1]
                else:
                    predict_y = torch.hstack([predict_y, torch.max(outputs, dim=1)[1]])
        return predict_y.cpu().numpy()
