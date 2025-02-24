import torch
from torch import nn
from quantization import *
from torch.nn import functional as F

class ResidualStack(nn.Module):
    def __init__(self,hidden_channels,residual_hidden,num_layers):
        super(ResidualStack,self).__init__()

        self.layers=nn.ModuleList()
        #这对么

        for _ in range(num_layers):
            block=nn.Sequential(nn.Conv2d(hidden_channels,residual_hidden,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(residual_hidden,hidden_channels,kernel_size=(1,1),stride=(1,1),bias=False))
            self.layers.append(block)


    def forward(self,x):

        for block in self.layers:
            b = block(x)
            x=x+b
        return x

class Encoder(nn.Module):
    def __init__(self,in_channels,hidden_channels,residual_hidden,num_layers):
        super(Encoder,self).__init__()

        self.enc1=nn.Conv2d(in_channels,hidden_channels//2,kernel_size=(4,4),stride=(2,2),padding=1)
        self.enc2=nn.Conv2d(hidden_channels//2,hidden_channels,kernel_size=(4,4),stride=(2,2),padding=1)
        self.enc3=nn.Conv2d(hidden_channels,hidden_channels,kernel_size=(3,3),stride=(1,1),padding=1)
        self.residual=ResidualStack(hidden_channels,residual_hidden,num_layers)
        self.relu=nn.ReLU()

    def forward(self,x):
        h=self.relu(self.enc1(x))
        h1=self.relu(self.enc2(h))
        h2=self.relu(self.enc3(h1))
        return self.residual(h2)



class Decoder(nn.Module):
    def __init__(self,in_channels,hidden_channels,embedding_dim,residual_hidden,num_layers):
        super(Decoder,self).__init__()

        self.dec1=nn.Conv2d(embedding_dim,hidden_channels,kernel_size=(3,3),stride=(1,1),padding=1)
        self.residual=ResidualStack(hidden_channels,residual_hidden,num_layers)
        self.dec2=nn.ConvTranspose2d(hidden_channels,hidden_channels//2,kernel_size=(4,4),stride=(2,2),padding=1)
        self.dec3=nn.ConvTranspose2d(hidden_channels//2,in_channels,kernel_size=(4,4),stride=(2,2),padding=1)
        self.relu=nn.ReLU()

    def forward(self,x):
        x=self.dec1(x)
        x=self.residual(x)
        x=self.relu(self.dec2(x))
        return self.dec3(x)




class VQVAE(nn.Module):
    def __init__(self,encoder,decoder,vqvae,pre_conv):
        super(VQVAE, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.vqvae=vqvae
        self.pre_conv=pre_conv

    def forward(self, x):
        z_e=self.pre_conv(self.encoder(x))
        z_q,encodings, quant_loss, vq_loss, commitment_loss=self.vqvae(z_e)
        recon=self.decoder(z_q)
        recon_loss=F.mse_loss(recon,x)#/variance????
        loss=recon_loss+quant_loss

        return recon,encodings,loss, recon_loss,quant_loss,vq_loss, commitment_loss

    def generate(self,x):
        return self.forward(x)






