import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from tqdm import tqdm    
from MRsiNet import MRSI3D
import torch
from torchmetrics import PeakSignalNoiseRatio 
from mrsi_utils import mrsi_train

from torchmetrics import StructuralSimilarityIndexMeasure as ssim
#from getPhantom_image_from_nii import data_rs as phantom_data
#from getInvivo_image_from_nii import data_rs as invivo_data
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mrsi_fs=torch.tensor(torch.load('mrsi_fs.pt')).to(device)
mrsi_us=torch.tensor(torch.load('mrsi_us.pt')).to(device)
#print(mrsi_fs.shape)
#exit()
mrsi_fs=mrsi_fs[...,0:4]
mrsi_us=mrsi_us[...,0:4]

lr=.001
epochs = 500# 100
batchsize=8#16#20
no_of_batches=50 # 500
N=batchsize*no_of_batches

model=MRSI3D().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()
psnr=PeakSignalNoiseRatio().to(device)
ssim=ssim().to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


snr=None
modelname=f"abs_mrsi_bs{batchsize}_bpe{no_of_batches}_snr{snr}_lr_{lr}"



modelboard=f"mrsi_runs/" + modelname
writer = SummaryWriter(modelboard)
model_dir="mrsi_models/"
modelpath = os.path.join(model_dir, modelname + '.pt')

if not os.path.exists(model_dir):
# Create a new directory if  does not exist
 os.makedirs(model_dir)
 print("The new directory is created!")


for epoch in tqdm(range(epochs)): 
    
    print(f" Epoch: {epoch+1}\n---------")
    train_loss= 0
    #model.train(True)
    #t1=time.time()
    for batch in tqdm(range(no_of_batches)):
                                 
            batch_fs, batch_us = next(mrsi_train(mrsi_fs=mrsi_fs, 
                                                 mrsi_us=mrsi_us, 
                                                 batchsize=batchsize,  
                                                 device=device, 
                                                 snr=None ) )
            #bs, 96 * 8, 22, 22, 21
            batch_us=batch_us.permute(3, 4, 5,  0, 1, 2)
            batch_us=batch_us.reshape(batchsize,96*8, 22, 22, 21 )
            #print(batch_fs.shape, batch_us.imag)
            batch_us_real = batch_us.real.to(dtype=torch.float32)
            batch_us_imag = batch_us.imag.to(dtype=torch.float32)
            #print(batch_us.shape)
            #exit()
            batch_rec = model(batch_us_real, batch_us_imag)
            batch_rec=batch_rec.reshape(batchsize, 96, 8, 22, 22, 21 )
            batch_rec=batch_rec.permute(3,4,5, 0, 1,2)
           
            #print(batch_rec.shape, batch_fs.shape)
            loss = criterion(batch_rec.abs(), batch_fs.abs())
            #loss=criterion(batch_rec.real, batch_fs.real) + criterion(batch_rec.imag, batch_fs.imag)
           
            # Calculate Loss
            train_loss += loss.item()

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

    # Calculate loss and accuracy per epoch 
    train_loss /= no_of_batches
    writer.add_scalar('Training Loss', train_loss, epoch+1)
    print(f"Train loss: {train_loss:.9} ")
    
    val_loss=0; psnr_value=0; str_sym=0
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        
        for batch in tqdm(range(no_of_batches)):
    
            batch_fs, batch_us = next(mrsi_train(mrsi_fs=mrsi_fs, 
                                                 mrsi_us=mrsi_us, 
                                                 batchsize=batchsize,  
                                                 device=device, 
                                                 snr=None ) )
            #bs, 96 * 8, 22, 22, 21
            batch_us=batch_us.permute(3, 4, 5,  0, 1, 2)
            batch_us=batch_us.reshape(batchsize,96*8, 22, 22, 21 )
            #print(batch_fs.shape, batch_us.imag)
            batch_us_real = batch_us.real.to(dtype=torch.float32)
            batch_us_imag = batch_us.imag.to(dtype=torch.float32)
            
            batch_rec = model(batch_us_real, batch_us_imag)
            batch_rec=batch_rec.reshape(batchsize, 96, 8, 22, 22, 21 )
            batch_rec=batch_rec.permute(3,4,5, 0, 1,2)
           
            #print(batch_rec.shape, batch_fs.shape)
            vloss = criterion(batch_rec.abs(), batch_fs.abs())
            #vloss=criterion(batch_rec.real, batch_fs.real) + criterion(batch_rec.imag, batch_fs.imag)
           
            psnr_value= psnr_value + psnr(batch_rec.abs().reshape(-1, 96,8,22,22,21), batch_fs.abs().reshape(-1, 96,8,22,22,21))
            #exit()
            #str_sym= str_sym+ ssim(batch_rec.abs().reshape(-1, 96*8,22,22,21), batch_fs.abs().reshape(-1, 96*8,22,22,21)) 
            #ncc = F.cross_correlation(reconstructed_batch, gtbatch)
            #ncc= ncc.item()
            val_loss += vloss.item()
                    
        # Adjust metrics and print out
        val_loss /= no_of_batches
        str_sym  /= no_of_batches
        psnr_value /= no_of_batches
        #if(epoch) % 10 == 9:
        torch.save(model.state_dict(), modelpath)
        writer.add_scalar('Validation Loss', val_loss, epoch+1)
        writer.add_scalar('PSNR', psnr_value, epoch+1)
        writer.add_scalar('Str. Symm.', str_sym, epoch+1)
        #writer.add_scalar('Normalized Cross. Corr.', ncc, epoch+1)       

        print(f"Val. loss: {val_loss:.9f} | PSNR: {psnr_value:.9} | Str.Symm.: {str_sym:.9}\n")



