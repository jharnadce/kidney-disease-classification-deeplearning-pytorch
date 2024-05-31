from pathlib import Path
import torch
from torch import nn
import torchvision
from cnnClassifierKidneyDisease.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig) -> None:
        self.config = config

    def get_base_model(self):
        self.model = torchvision.models.vgg16()
        # Remove the last set of dense layers
        if self.config.params_include_top == False:
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        self.save_model(model=self.model, path = self.config.base_model_path)

    @staticmethod
    def _prepare_full_model(model, classes, learning_rate):
        # Freeze the parameters of the pre-trained VGG16 layers
        for params in model.parameters():
                params.requires_grad = False

        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

        # Define the custom classifier
        classifier = nn.Sequential(
            nn.Flatten(),
            # nn.Dropout(0.2),
            # nn.Linear(512 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, classes),
        )

        full_model = nn.Sequential(model, classifier).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(full_model.parameters(), lr=learning_rate)

        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_num_classes,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(model=self.full_model, path=self.config.updated_base_model_path)        
            

    @staticmethod
    def save_model(model, path: Path):
        torch.save(model, path)
