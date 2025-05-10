import argparse
import json
import models
import training_loss

import torch
from dataset_manager import GenerativeAIDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformation.data_preprocessing import get_data_composing

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train-dataset", type=str, default='../data/train', help='path of train dataset')
parser.add_argument("--expand-dataset", type=int, default=100, help='Repeat data')
parser.add_argument("--batch-size", type=int, default=8, help='batch size')
parser.add_argument("--learning-rate", type=float, default=1e-3, help='learning rate for optimization')
parser.add_argument("--lr-scheduler-patience", type=int, default=5, help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
parser.add_argument("--lr-scheduler-gamma", type=float, default=0.1, help='learning rate is multiplied by the gamma to decrease it')
parser.add_argument("--max-epochs", type=int, default=70, help='max number of epochs')
parser.add_argument("--model", choices=models.available_models, default=models.available_models[0], help='model of GEN AI')
parser.add_argument("--config", type=str, default='./config.json', help='path of difussion config')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = json.load(f)


device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = args.batch_size
model_name = args.model
lr = args.learning_rate

data_processing = get_data_composing(model_name, config)
train_dataset = GenerativeAIDataset(args.train_dataset,
                            data_processing,
                            class_c='BOAFAB',
                            expand_data=args.expand_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True)

model = models.create_model(model_name, config=config).to(device)
training_loss_function = training_loss.get_loss_train(model_name, general_config=config)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                            patience=args.lr_scheduler_patience,
                                            factor=args.lr_scheduler_gamma)


def train(epoch):
    
    model.train()  # Set model to training mode (lo dejo si hago validación)

    running_loss = 0.0
    pbar = tqdm(train_dataloader,
            unit="audios",
            unit_scale=train_dataloader.batch_size,
            desc=f'Epoca {epoch}')
    for step, batch in enumerate(pbar):
        batch = batch['samples'].to(device)
        optimizer.zero_grad()
        loss = training_loss_function(model, batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / (step + 1))
        })

    loss_epoch = running_loss / len(train_dataloader)

    return loss_epoch

print(f'Training {model_name}')

for epoch in range(args.max_epochs):

    loss_train = train(epoch)

    lr_scheduler.step(metrics=loss_train)
    
    if epoch % 100 == 0:
        if model_name == 'VAE':
            path = f'./models_trained/VAE/{model_name}_model.pth'
        else:
            path = f'./models_trained/DIFFUSION/{epoch}_{model_name}_model.pth'

        torch.save(model, path)
   

torch.save(model, f'{model_name}_model.pth')

print('Model was save')