import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from tqdm import tqdm    
from MRsiNet import MRSI3D, MRSI3D_WB
import torch
from torchmetrics import PeakSignalNoiseRatio 
from mrsi_utils import mrsi_train, mrsi_train2

from torchmetrics import StructuralSimilarityIndexMeasure as ssim
#from getPhantom_image_from_nii import data_rs as phantom_data
#from getInvivo_image_from_nii import data_rs as invivo_data
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


mrsi_fs=torch.tensor(torch.load('lowranked_mrsi_fs.pt'))
mrsi_us=torch.tensor(torch.load('lowranked_mrsi_us.pt'))
mask=torch.tensor(torch.load('numpy_mask.pt'))


## Here to use FFT over FID, to train on spectrum instead of FID
mrsi_us=torch.tensor(np.fft.fftshift(np.fft.fft(mrsi_us, axis=-3), axes=-3))
mrsi_fs=torch.tensor(np.fft.fftshift(np.fft.fft(mrsi_fs, axis=-3), axes=-3))

#print(mrsi_fs.shape)
#exit()

# Select first 5 subjects for training, and last for testing!
mrsi_fs=mrsi_fs.to(torch.complex64)
mrsi_us=mrsi_us.to(torch.complex64)

mrsi_fs=mrsi_fs*mask[:,:,:, None, None,:]
mrsi_us=mrsi_us*mask[:,:,:, None, None,:]


#mrsi_fs_train = np.delete(mrsi_fs, 5, axis=-1)
#mrsi_us_train = np.delete(mrsi_us, 5, axis=-1)  

norm_mrsi_fs_train=torch.zeros(22,22,21,96,8,5, dtype=torch.complex64)
norm_mrsi_us_train=torch.zeros(22,22,21,96,8,5, dtype=torch.complex64)
for i in range(5):
 norm_mrsi_fs_train[..., i]  = mrsi_fs[..., i]/ mrsi_fs[..., i].abs().max()
 norm_mrsi_us_train[..., i]  = mrsi_us[..., i]/ mrsi_us[..., i].abs().max()

# Normalize training and test data using train_max
#norm_mrsi_fs_train = mrsi_fs_train/ mrsi_fs_train.abs().max()
#norm_mrsi_us_train = mrsi_us_train/ mrsi_us_train.abs().max()

#norm_mrsi_fs_test = mrsi_fs_test / mrsi_fs_train.abs().max()
#norm_mrsi_us_test = mrsi_us_test / mrsi_us_train.abs().max()


mrsi_fs=norm_mrsi_fs_train.to(device)
mrsi_us=norm_mrsi_us_train.to(device)

lr=.0001
epochs = 10000# 100
batchsize=500#8, 4
no_of_batches=2 # 500
N=batchsize*no_of_batches

model=MRSI3D_WB().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()
psnr=PeakSignalNoiseRatio().to(device)
ssim=ssim().to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

snr=None
modelname=f"FFT_corrRots_Norm_LR_mrsi_bs{batchsize}_bpe{no_of_batches}_snr{snr}_lr_{lr}"

modelboard=f"mrsi_runs/" + modelname
writer = SummaryWriter(modelboard)
model_dir="mrsi_models/"
modelpath = os.path.join(model_dir, modelname + '.pt')

if not os.path.exists(model_dir):
# Create a new directory if  does not exist
 os.makedirs(model_dir)
 print("The new directory is created!")

#torch.cuda.empty_cache()

for epoch in tqdm(range(epochs)): 
    
    print(f" Epoch: {epoch+1}\n---------")
    train_loss= 0
    #model.train(True)
    #t1=time.time()
    for batch in tqdm(range(no_of_batches)):
                                 
            batch_fs, batch_us = next(mrsi_train(mrsi_fs=mrsi_fs, 
                                                 mrsi_us=mrsi_us,
                                                 batchsize=batchsize, 
                                                 device=device) )
            #print(batch_fs.shape, batch_fs.dtype)
            #print(batch_us.shape, batch_us.dtype)
            #exit()
            
            batch_us=batch_us.to(device)
            batch_fs=batch_fs.to(device)
            batch_rec = model(batch_us.real, batch_us.imag )
            
            #print(batch_rec.shape, batch_rec.dtype)
            #exit()
            loss=criterion(batch_rec.real, batch_fs.real) + criterion(batch_rec.imag, batch_fs.imag)
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
                                                 device=device) )
            #print(batch_fs.shape, batch_fs.dtype)
            #print(batch_us.shape, batch_us.dtype)
            #exit()
            batch_us=batch_us.to(device)
            batch_fs=batch_fs.to(device)
            
            batch_rec = model(batch_us.real, batch_us.imag )
            
            #print(batch_rec.shape, batch_rec.dtype)
            #print(batch_rec.shape, batch_rec.dtype)
            #exit()
            vloss=criterion(batch_rec.real, batch_fs.real) + criterion(batch_rec.imag, batch_fs.imag)
            
           
            psnr_value= psnr_value + psnr(batch_rec.abs(), batch_fs.abs())
            #print(batch_rec.abs().shape) 
            #psnr_value= psnr_value + .5*psnr(batch_rec.real, batch_fs.real) + \
             #                        .5*psnr(batch_rec.imag, batch_fs.imag)
           
            #exit()
            #str_sym= str_sym + 5*ssim(batch_rec.real, batch_fs.real) + \
            #                   .5*ssim(batch_rec.imag, batch_fs.imag)
            str_sym= str_sym + ssim(batch_rec.abs(), batch_fs.abs()) 
            
           
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


