import torch
import pandas as pd
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fake_ecg(model, size, noise_dim, save_path):
    
    df_fake = pd.DataFrame({thing:[] for thing in ['ecg', 'label']})

    for thing in ['ecg']:
                df_fake[thing] = df_fake[thing].astype(object)
    
    class_list = [1,2,3,3,4,4,5,6,7,8,9,9]

    for i in range(size):
        fixed_noise_gen = torch.randn(1, 12, noise_dim).to(device)
        # random_num = random.randint(0,9)
        random_num = random.choice(class_list)
        label_fake = torch.ones(1, dtype = int).to(device)
        label_fake  = label_fake * random_num
        fake_ecg = model(fixed_noise_gen, label_fake)
        fake_ecg = fake_ecg.cpu().detach().numpy()

        df_fake.at[i, 'ecg'] = fake_ecg
        df_fake.at[i, 'label'] = int(random_num)
    
    df_fake['ecg'] = df_fake['ecg'].apply(lambda x: np.squeeze(x,0))
    df_fake.to_pickle(save_path)

### WGAN-GP BASE ###

# model_path_wgan_base = '/home/danielsantosh/data_augmentation/models/WGAN_gen_base_200e.pt'
# gen_wgan_base = torch.load(model_path_wgan_base)

# path_save_WGAN_base_200e = '/home/danielsantosh/data_augmentation/fake_ecg/fake_ecg_WGAN_base_200e_classimb.pkl'
# fake_ecg(gen_wgan_base, 1000, 624, path_save_WGAN_base_200e)

# ### WGAN-GP OPTIM1 ###

# model_path_wgan_optim1 = '/home/danielsantosh/data_augmentation/models/WGAN_gen_optim1_500e.pt'
# gen_wgan_optim1 = torch.load(model_path_wgan_optim1)

# path_save_WGAN_optim1_500e = '/home/danielsantosh/data_augmentation/fake_ecg/fake_ecg_WGAN_optim1_500e.pkl'
# fake_ecg(gen_wgan_optim1, 1000, 624, path_save_WGAN_optim1_500e)

# ### DCGAN BASE ###

model_path_dcgan_base = '/home/danielsantosh/data_augmentation/models/DCGAN_gen_base_200e.pt'
gen_dcgan_base = torch.load(model_path_dcgan_base)

path_save_DCGAN_base_200e = '/home/danielsantosh/data_augmentation/fake_ecg/fake_ecg_DCGAN_base_200e_classimb_triple.pkl'
fake_ecg(gen_dcgan_base, 3000, 624, path_save_DCGAN_base_200e)





