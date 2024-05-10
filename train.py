import lightning as pl
from torch import nn
import torch
from datamodule import SimpleDataModule  # Adjust import according to actual path and names
from models import FNO1DEncoder, FNO1DDecoder  # Adjust import according to actual path and names
from rwkv_models import RWKV_model  # Adjust import according to actual path and names
from dataclasses import dataclass
from lightning.pytorch.loggers import TensorBoardLogger

def nrmse_loss(output, target):
    # Assuming target is not all the same values, which would make the denominator zero.
    loss = torch.sqrt(torch.mean((output - target) ** 2))
    norm_factor = target.max() - target.min()
    return loss / norm_factor if norm_factor != 0 else loss


class OmniArch_fourier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = FNO1DEncoder(
            modes1 = config.modes1,
            width = config.width,
            emb_dim = config.n_embd
        )

        self.rwkv = RWKV_model(
            config
        )

        self.decoder = FNO1DDecoder(
            modes1 = config.modes1,
            width= config.width,
            emb_dim = config.n_embd
        )

        self.loss_fn = nn.MSELoss()
    
    def forward(self, x, grid):
        '''
        '''
        
        bsz, ts, grid_size = x.shape

        tokens = self.encoder(x.reshape(bsz*ts, grid_size), grid) #(bsz*seq_len, dim)
        latent = self.rwkv(tokens.reshape(bsz, ts, -1)) 
        
        # print("latent shape:{}".format(latent.shape))
        # latent should be batchify along the seq_len dim

        latent = latent.reshape(bsz*ts, -1)
        
        output = self.decoder(latent, x_size=(bsz*ts, self.config.width, grid_size + 2))

        output = output.reshape(bsz, ts, grid_size)

        return output
    
    def training_step(self, batch, batch_idx):
        x, grid = batch

        target = torch.clone(x)

        x = x[:, :-1, :]
        target = target[:, 1:, :]
        bsz, ts, grid_size = x.shape
        # Assuming x is of the shape [batch, timesteps, features]
        # We need to create a shifted version of x for target
        

        # x = x[:, :-1, :]
        grid = grid.unsqueeze(1).repeat(1, ts, 1).reshape(bsz*ts, grid_size)
        predictions = self.forward(x, grid)
        # Loss calculation: compare all but the last timestep (since it has no "next" timestep)
        loss = self.loss_fn(predictions[:, :-1, :], target[:, 1:, :])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, grid = batch
        target = torch.clone(x)

        bsz, grid_size = grid.size(0), grid.size(1)
         # Autoregressive predictions
        current_input = x[:, 0:1, :]  # Start with the first timestep
        # predictions = []

        
        for t in range(1, x.size(1)):  # Assume x.size(1) is the number of timesteps
            
            current_grid = grid.unsqueeze(1).repeat(1, current_input.size(1), 1).reshape(bsz*t, grid_size)
            # batch_input = current_input.reshape(bsz*t, grid_size)
            prediction = self.forward(current_input, current_grid)[:, -1, :]  # Predict next step
            # predictions.append(prediction)
            # prediction = prediction
            current_input = torch.cat((current_input, prediction.unsqueeze(1)), dim=1)  # Use the prediction as the next input
        
        # import pudb;pu.db;
        # Calculate NRMSE for the last timestep
        last_pred = prediction
        last_true = target[:, -1, :]
        loss = nrmse_loss(last_pred, last_true)
        self.log('val_nrmse_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

@dataclass
class OmniArchConfig:
    block_size: int = 200
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better
    width: int = 64
    modes1: int = 16


# Initialize DataModule
data_module = SimpleDataModule(data_dir='/mnt/f/data/PDEbench/1D/Advection/Train', batch_size=2)


config = OmniArchConfig()

# Initialize Model
model = OmniArch_fourier(config)

# Initialize Trainer
trainer = pl.Trainer(max_epochs=10, logger=TensorBoardLogger('lightning_logs/'))
trainer.fit(model, datamodule=data_module)