import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib.gridspec import GridSpec
import numpy as np
from tqdm import tqdm    
from MRsiNet import MRSI3D_WB
import torch
from tensorboardX import SummaryWriter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from mrsi_utils import nmse, plot_mrsi_spectra



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mrsi_fs=torch.tensor(torch.load('lowranked_mrsi_fs.pt'))
mrsi_us=torch.tensor(torch.load('lowranked_mrsi_us.pt'))
mask=torch.tensor(torch.load('numpy_mask.pt'))

#print(mrsi_fs.shape)
#exit()

# Select first 5 subjects for training, and last for testing!
mrsi_fs=mrsi_fs.to(torch.complex64)
mrsi_us=mrsi_us.to(torch.complex64)

## Here to use FFT over FID, to train on spectrum instead of FID
#mrsi_us=torch.tensor(np.fft.fftshift(np.fft.fft(mrsi_us, axis=-3), axes=-3))
#mrsi_fs=torch.tensor(np.fft.fftshift(np.fft.fft(mrsi_fs, axis=-3), axes=-3))

mrsi_fs=mrsi_fs*mask[:,:,:, None, None,:]
mrsi_us=mrsi_us*mask[:,:,:, None, None,:]


mrsi_fs_test = mrsi_fs[..., 5] 
mrsi_us_test = mrsi_us[..., 5] 

norm_mrsi_fs_test = mrsi_fs_test/ mrsi_fs_test.abs().max()
norm_mrsi_us_test = mrsi_us_test/ mrsi_us_test.abs().max()


mrsi_fs=norm_mrsi_fs_test.to(device)
mrsi_us=norm_mrsi_us_test.to(device)

model=MRSI3D_WB().to(device)
batchsize=1
snr=None

modelname=f"corrRots_Norm_LR_mrsi_bs500_bpe2_snrNone_lr_0.0001"


model.load_state_dict(torch.load(os.path.join("mrsi_models//",  modelname + '.pt')))
#model=torch.load(os.path.join("mrsi_models//",  modelname + '.pt'))

mrsi_recons=torch.zeros(22, 22, 21, 96, 8, dtype=torch.complex64)
#print(mrsi_us.real[ :,:,:, 0,0].squeeze(-1).shape)

model.eval() 
with torch.inference_mode(): 
        for t in range(96):
            for T in range(8):
                mrsi_recons[:,:,:, t,T] = model(mrsi_us.real[None, :,:,:, t,T], mrsi_us.imag[None, :,:,:, t,T]) 

torch.save(mrsi_recons, "SK_mrsi_recons.pt")
#mrsi_recons=mrsi_recons*mask[:,:,:,0, None, None].cpu()
#im=torch.load( "norm_mrsi_recons.pt")
#mrsi_recons=mrsi_recons/mrsi_recons.abs().max()

mrsi_fs=mrsi_fs.cpu().numpy().squeeze()
mrsi_us=mrsi_us.cpu().numpy().squeeze()
mrsi_recons=mrsi_recons.detach().numpy().squeeze()
#print(mrsi_fs.shape, mrsi_fs.dtype)


#plt.imshow(np.abs(mrsi_fs[:,:,10,50,7]))
#plt.show()
            
# Parameters
slice_index = 10  # 10th slice (zero-based indexing)
print(mrsi_fs.shape,mrsi_fs[10, 10, 10, 10, 7].dtype)

mrsi_us1=np.abs(np.fft.fftshift(np.fft.fft(mrsi_us, axis=-2), axes=-2))
mrsi_fs1=np.abs(np.fft.fftshift(np.fft.fft(mrsi_fs, axis=-2), axes=-2))
mrsi_recons1=np.abs(np.fft.fftshift(np.fft.fft(mrsi_recons, axis=-2), axes=-2))

#plt.imshow((mrsi_fs1[:, :, 10, 50, 7]))
#plt.show()

#i, j, k = 10, 10, 10
i, j, k = 13, 10, 10

plot_mrsi_spectra(mrsi_fs1, mrsi_us1, mrsi_recons1, i=i, j=j, k=k)


T=8
t=50
us_images = mrsi_us1[:, :, slice_index, t, :T]
fs_images = mrsi_fs1[:, :, slice_index, t, :T]
recons_images = mrsi_recons1[:, :, slice_index, t, :T]

ssim_value1 = ssim(recons_images, fs_images, data_range=fs_images.max() - fs_images.min())
psnr_value1 = psnr(recons_images, fs_images, data_range=fs_images.max() - fs_images.min())
nmse_value1 = nmse(recons_images, fs_images)
print(mrsi_fs.dtype)

print("psnr on T dim=", psnr_value1, "ssim on T dim=",ssim_value1, "NMSE on T dim=", nmse_value1)




#exit()
# Create a grid of 3x8 images
vmin = fs_images.min()#np.percentile(fs_images, 1)  # Lower 1% percentile
vmax = fs_images.max()#np.percentile(fs_images, 99) # 
fig = plt.figure(figsize=(16, 6))
gs = GridSpec(3, 9, figure=fig, width_ratios=[0.3] + [1]*8)
row_labels = ["Under Sampled", "Ground Truth", "Reconstructed"]

for i in range(3):
    # Add row labels
    ax_label = fig.add_subplot(gs[i, 0])
    ax_label.text(1.2, 0.5, row_labels[i], rotation='vertical', fontsize=12, va='center', ha='right',transform=ax_label.transAxes)
    ax_label.axis('off')

    for j in range(8):
        ax = fig.add_subplot(gs[i, j + 1])
        if i==0:
         vmin = np.percentile(fs_images[:, :, j], 1)
         vmax = np.percentile(fs_images[:, :, j], 99)
         
         ax.imshow(us_images[:, :, j], vmin=vmin, vmax=vmax)

         fs_images1=fs_images[:,:,j]
         us_images1=us_images[:,:,j]
         ssim_value = ssim(fs_images1, us_images1, data_range=fs_images1.max() - fs_images1.min())
         psnr_value = psnr(fs_images1, us_images1, data_range=fs_images1.max() - fs_images1.min())
         
         nmse_value1 = nmse(fs_images1, us_images1)
         ax.set_title(f"psnr={psnr_value:.2f}, ssim={ssim_value:.2f} , NMSE={nmse_value1:.2f}", fontsize=6 )
         ax.axis('off')
         
        if i==1:
         #vmin = fs_images[:, :, j].min()
         #vmax = 0.9*fs_images[:, :, j].max()
         vmin = np.percentile(fs_images[:, :, j], 1)
         vmax = np.percentile(fs_images[:, :, j], 99)
        
         ax.imshow(fs_images[:, :, j], vmin=vmin, vmax=vmax)
         ax.axis('off')
        
        if i==2:
         #vmin = fs_images[:, :, j].min()
         #vmax = 0.9*fs_images[:, :, j].max()
         vmin = np.percentile(fs_images[:, :, j], 1)
         vmax = np.percentile(fs_images[:, :, j], 99)
        
         ax.imshow(recons_images[:, :, j], vmin=vmin, vmax=vmax)

         fs_images1=fs_images[:,:,j]
         recons_images1=recons_images[:,:,j]
         
         ssim_value = ssim(fs_images1, recons_images1, data_range=fs_images1.max() - fs_images1.min())
         psnr_value = psnr(fs_images1, recons_images1, data_range=fs_images1.max() - fs_images1.min())
         
         nmse_value1 = nmse(fs_images1, recons_images1)
         ax.set_title(f"psnr={psnr_value:.2f}, ssim={ssim_value:.2f} , NMSE={nmse_value1:.2f}", fontsize=6 )

         
         ax.axis('off')

plt.suptitle(f"t-slice={t}, z-slice {slice_index}")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
 
 
#...............................T slice...................
T=7
t=8
us_images = mrsi_us1[:,:,slice_index,  45:53, T]
fs_images = mrsi_fs1[:,:,slice_index,  45:53, T]
recons_images = mrsi_recons1[:,:,slice_index,  45:53, T]
print(us_images.shape, fs_images.shape)
ssim_value1 = ssim(recons_images, fs_images, data_range=fs_images.max() - fs_images.min())
psnr_value1 = psnr(recons_images, fs_images, data_range=fs_images.max() - fs_images.min())
nmse_value1 = nmse(mrsi_recons[:, :, slice_index, t, :T], mrsi_fs[:, :, slice_index, t, :T])
print(mrsi_fs.dtype)

print("psnr on T dim=", psnr_value1, "ssim on T dim=",ssim_value1, "NMSE on T dim=", nmse_value1)

#exit()         
# Create a grid of 3x8 images
fig = plt.figure(figsize=(16, 6))
gs = GridSpec(3, 9, figure=fig, width_ratios=[0.3] + [1]*8)
row_labels = ["Under Sampled", "Ground Truth", "Reconstructed"]

for i in range(3):
    # Add row labels
    ax_label = fig.add_subplot(gs[i, 0])
    ax_label.text(1.2, 0.5, row_labels[i], rotation='vertical', fontsize=12, va='center', ha='center')
    ax_label.axis('off')

    for j in range(8):
        ax = fig.add_subplot(gs[i, j + 1])
        if i==0:
         ax.imshow(us_images[:, :, j])
         fs_images1=fs_images[:,:,j]
         us_images1=us_images[:,:,j]
         ssim_value = ssim(fs_images1, us_images1, data_range=fs_images1.max() - fs_images1.min())
         psnr_value = psnr(fs_images1, us_images1, data_range=fs_images1.max() - fs_images1.min())
         ax.set_title(f"psnr={psnr_value:.2f}, ssim={ssim_value:.2f}",fontsize=10 )
         ax.axis('off')
         
        if i==1:
         ax.imshow(fs_images[:, :, j])
         ax.axis('off')
        if i==2:
         
         ax.imshow(recons_images[:, :, j])
         fs_images_=fs_images[:,:,j]
         #us_images_=us_images[:,:,j]
         recons_images_=recons_images[:,:,j]
         ssim_value = ssim(recons_images_, fs_images_, data_range=fs_images_.max() - fs_images_.min())
         psnr_value = psnr(recons_images_, fs_images_, data_range=fs_images_.max() - fs_images_.min())
         #ssim_value = ssim(us_images_, fs_images_, data_range=fs_images.max() - fs_images.min())
         #psnr_value = psnr(us_images_, fs_images_, data_range=fs_images.max() - fs_images.min())
         
         ax.set_title(f"psnr={psnr_value:.2f}, ssim={ssim_value:.2f}",fontsize=10 )
         ax.axis('off')

plt.suptitle(f"T-slice={T}, z-slice 10")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
