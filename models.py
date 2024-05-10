import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class FNO1DEncoder(nn.Module):
  def __init__(self, modes1=16, width=64, emb_dim=1024):
    super(FNO1DEncoder, self).__init__()
    self.modes1 = modes1
    self.width = width
    self.emb_dim = emb_dim 
    
    self.head = nn.Linear(width*modes1*2, self.emb_dim)
    self.padding = 2
    self.fc0 = nn.Linear(2, self.width) # input channel is 2:(a(x), x)

    self.in_channels = self.width
    self.out_channels = self.width
    self.scale = 1
    self.scale = (1 / (self.in_channels * self.out_channels))
    self.weights1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, dtype=torch.cfloat))
  
  def compl_mul1d(self, input, weights):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", input, weights)
  
  def forward(self, x, grid):
    x = torch.cat((x.unsqueeze(-1), grid.unsqueeze(-1)), dim=-1) #[bsz, seq, 2]
    x = self.fc0(x) #[bsz, seq, width]

    x = x.permute(0, 2, 1) #[bsz, width, seq]

    x = F.pad(x, [0, self.padding])

    batchsize = x.shape[0]
    #Compute Fourier coeffcients up to factor of e^(- something constant)
    x_ft = torch.fft.rfft(x)
    # Multiply relevant Fourier modes 
    out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)

    out_ft =  self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

    out = torch.cat([out_ft.real, out_ft.imag], dim=-1)

    out = self.head(out.view(batchsize, -1))

    return out 

class FNO1DDecoder(nn.Module):
  def __init__(self, modes1=16, width=64, emb_dim=1024):
    super(FNO1DDecoder, self).__init__()
    self.modes1 = modes1
    self.width = width
    self.emb_dim = emb_dim 
    
    self.decode_head = nn.Linear(self.emb_dim, width*modes1*2)

    self.fc1 = nn.Linear(self.width, 128)
    self.fc2 = nn.Linear(128, 1)
    self.padding = 2
    self.fc0 = nn.Linear(2, self.width) # input channel is 2:(a(x), x)

    self.in_channels = self.width
    self.out_channels = self.width
    self.scale = 1
    self.scale = (1 / (self.in_channels * self.out_channels))
  
  def forward(self, token, x_size):
    # token shape (bsz, embed_dim)
    bsz = token.size(0)
    hidden_states = self.decode_head(token)
    out_ft_modes = hidden_states.reshape(bsz, self.out_channels, self.modes1, 2)
    
    out_ft_modes = torch.complex(out_ft_modes[:,:,:,0], out_ft_modes[:,:,:,1])

    out_ft = torch.zeros(bsz, self.out_channels,  x_size[-1]//2 + 1,  device=out_ft_modes.device, dtype=torch.cfloat)
    out_ft[:,:,:self.modes1] = out_ft_modes

    x_ = torch.fft.irfft(out_ft, n=x_size[-1])

    x = x_[..., :-self.padding]
    x = x.permute(0, 2, 1)

    x = self.fc1(x)
    x = torch.nn.functional.gelu(x)
    x = self.fc2(x)
    # print("after fc2", x.shape)
    return x.squeeze(-2)
  

class OmniArch_RWKV(nn.Module):
  def __init__(self, encoder, rwkv, decoder):
    super(OmniArch_RWKV, self).__init__()
    self.encoder = encoder
    self.rwkv = rwkv
    self.decoder = decoder
  
  def forward(self, x, grid):
    '''
    x: shape (bsz, timesteps, grid_size)
    grid: shape (bsz, grid_size)
    '''
    bsz, ts, grid_size = x.shape

    tokens = self.encoder(x.reshape(bsz*ts, grid_size), grid) #(bsz*seq_len, dim)


    latent = self.rwkv(tokens.reshape(bsz, ts, -1)) 

    decode_physics = self.decoder(latent, x_size=(bsz, width, grid_size + 2))

    return decode_physics




if __name__ == "__main__":
  # 1. Test the FNO1D encoder and decoder
  
  test_grid_size = 1024
  width = 64
  bsz = 4
  padding = 2


  encoder1D = FNO1DEncoder() # default token size: 1024
  decoder1D = FNO1DDecoder() # default token size: 1024


  test_input_x = torch.randn(bsz, test_grid_size, 1)
  test_grid = torch.randn(bsz, test_grid_size, 1)

  token = encoder1D(test_input_x, test_grid)

  print(token.shape) # torch.Size([4, 1024])

  
  decode_out = decoder1D(token, x_size=(bsz, width, test_grid_size + 2))

  print(decode_out.shape) # torch.Size([4, 1024])
  

