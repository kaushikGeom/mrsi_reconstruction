import torch
import numpy as np
import random


def extract_right_part(vector):
    right_parts = []
    for element in vector:
        right_part = element.split('_')[1]
        right_parts.append(right_part)
    return [int(right) for right in right_parts]


def vector(ind, total=None):
    result_vector = []
    for num in range(total):
        random_index = random.randint(2, ind)
        result_vector.append(f"{num}_{random_index}")
    #print("\nResulting vector:", result_vector)
    return extract_right_part(result_vector)


def mrsi_train(mrsi_fs=None,  mrsi_us=None,  batchsize=None, device=None, snr=None ):
            
      sizet=mrsi_fs.shape[3]
      sizeT=mrsi_fs.shape[4]
      patients_count=mrsi_fs.shape[5]
      #vectort=np.array(vector(patients_count, total=4),dtype=int)
      #vectorT=np.array(vector(patients_count, total=5),dtype=int)
      #mrsi_fs=mrsi_fs.to(device)
      #mrsi_us=mrsi_us.to(device)

      batch_fs=torch.zeros(22, 22, 21, batchsize, 96, 8, dtype=mrsi_fs.dtype, device=device)    
      batch_us=torch.zeros(22, 22, 21, batchsize, 96, 8, dtype=mrsi_us.dtype, device=device)       
      #print(mrsi_fs.shape)
      for b in range(batchsize):  
            for t in range(96):
                for T in range(8):
                    indst=random.randint(0, 3)
                    indsT=random.randint(0, 3)
                    batch_fs[..., b,  t, T]= mrsi_fs[:,:,:, t, T, indst] 
                    batch_us[..., b,  t, T]= mrsi_us[:,:,:, t, T, indst] 
                    
      yield batch_fs, batch_us

