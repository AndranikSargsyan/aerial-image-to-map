#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from qartezator.data.dataset import QartezatorDataset
from qartezator.data.datamodule import QartezatorDataModule
from qartezator.data.transforms import get_common_augmentations
import numpy as np
from PIL import Image
import torch
from torch import nn
import matplotlib.pyplot as plt
from helper import save_models
from helper import update_image_pool
from cyclegan.model import define_discriminator
from cyclegan.model import resnet_block
from cyclegan.model import define_generator
from cyclegan.model import define_composite_model


# In[ ]:


root_path = './data/maps'
train_txt_path = './assets/train.txt'
val_txt_path = './assets/val.txt'
test_txt_path = './assets/test.txt'


# In[ ]:


ds = QartezatorDataset(
    root_path=root_path,
    split_file_path=train_txt_path,
    common_transform=get_common_augmentations(256)
)
sample_source_img, sample_target_img = ds[42]


# In[ ]:


image_uint8 = (sample_source_img * 255).astype(np.uint8)

# Create the PIL Image object
source_image = Image.fromarray(image_uint8)


# In[ ]:


source_image


# In[ ]:


image_uint8 = (sample_target_img * 255).astype(np.uint8)

# Create the PIL Image object
image_pil = Image.fromarray(image_uint8)


# In[ ]:


image_pil


# In[ ]:


dm = QartezatorDataModule(
    root_path=root_path,
    train_txt_path=train_txt_path,
    val_txt_path=val_txt_path,
    test_txt_path=test_txt_path,
    input_size=256,
    train_batch_size = 32,
    val_batch_size = 32,
    test_batch_size= 32
)
train_dataloader = dm.train_dataloader()
val_dataloader = dm.val_dataloader()
test_dataloader = dm.test_dataloader()


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, train_dataloader, epochs=1):
    # Define properties of the training run
    n_epochs= epochs
    n_batch=32
    # Determine the output square shape of the discriminator
    n_patch = 324
    
    # Prepare image pool for fake images
    poolA, poolB = [], []
    # Calculate the number of batches per training epoch
    bat_per_epo = len(train_dataloader) // n_batch
    # Calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    print(n_steps)
    # Define loss functions
    adversarial_loss = nn.MSELoss()
    identity_loss = nn.L1Loss()
    cycle_loss = nn.L1Loss()

    # Define optimizers
    optimizer_c_AtoB = optim.Adam(c_model_AtoB.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_c_BtoA = optim.Adam(c_model_BtoA.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_d_A = optim.Adam(d_model_A.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_d_B = optim.Adam(d_model_B.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_g_AtoB = optim.Adam(g_model_AtoB.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_g_BtoA = optim.Adam(g_model_BtoA.parameters(), lr=0.001, betas=(0.5, 0.999))

   

    # Manually enumerate epochs
    for epoch in range(n_epochs):
        # Enumerate over the data loaders
        for i, (real_B,real_A) in enumerate(train_dataloader):
            # Move real images to the device
            real_A=(real_A-0.5)/0.5
            real_B=(real_B-0.5)/0.5
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            

            ##############################
            # Update AtoB Generator
            ##############################

            # Set generators' gradients to zero
            optimizer_g_AtoB.zero_grad()
            optimizer_c_AtoB.zero_grad()

            # Generate fake images
            fake_B = g_model_AtoB(real_A)
            fake_A = g_model_BtoA(real_B)
            
            # Update image pool for fake images
            fake_A = update_image_pool(poolA, fake_A.detach().cpu().numpy())
            fake_B = update_image_pool(poolB, fake_B.detach().cpu().numpy())
            fake_A_tensor = torch.from_numpy(fake_A).to(device)
            fake_B_tensor = torch.from_numpy(fake_B).to(device)

            # Adversarial loss
            
            pred_fake_A = d_model_A(fake_A_tensor)
            pred_fake_B = d_model_B(fake_B_tensor)
            loss_adv_AtoB = adversarial_loss(pred_fake_A, torch.ones_like(pred_fake_A))
            loss_adv_BtoA = adversarial_loss(pred_fake_B, torch.ones_like(pred_fake_B))

            # Identity loss
            idt_A = g_model_BtoA(real_A)
            idt_B = g_model_AtoB(real_B)
            loss_idt_A = identity_loss(idt_A, real_A)
            loss_idt_B = identity_loss(idt_B, real_B)

            # Cycle-consistency loss
            
            cycle_A = g_model_BtoA(fake_B_tensor)
            cycle_B = g_model_AtoB(fake_A_tensor)
            loss_cycle_A = cycle_loss(cycle_A, real_A)
            loss_cycle_B = cycle_loss(cycle_B, real_B)

            # Total loss
            loss_g_AtoB = loss_adv_AtoB + 5 * loss_idt_A + 10 * loss_cycle_A
            loss_g_BtoA = loss_adv_BtoA + 5 * loss_idt_B + 10 * loss_cycle_B

            # Backpropagation and optimization
            loss_g_AtoB.backward()
            optimizer_g_AtoB.step()
            optimizer_c_AtoB.step()

            ##############################
            # Update BtoA Generator
            ##############################

            # Set generators' gradients to zero
            optimizer_g_BtoA.zero_grad()
            optimizer_c_BtoA.zero_grad()

            # Generate fake images
            fake_A = g_model_BtoA(real_B)
            fake_B = g_model_AtoB(real_A)

            # Update image pool for fake images
            fake_B = update_image_pool(poolB, fake_B.detach().cpu().numpy())
            fake_A = update_image_pool(poolA, fake_A.detach().cpu().numpy())
            fake_A_tensor = torch.from_numpy(fake_A).to(device)
            fake_B_tensor = torch.from_numpy(fake_B).to(device)

            # Adversarial loss
            #fake_A_tensor = torch.from_numpy(fake_A)
            #fake_B_tensor = torch.from_numpy(fake_B)
            pred_fake_B = d_model_B(fake_B_tensor)
            pred_fake_A = d_model_A(fake_A_tensor)
            loss_adv_BtoA = adversarial_loss(pred_fake_B, torch.ones_like(pred_fake_B))
            loss_adv_AtoB = adversarial_loss(pred_fake_A, torch.ones_like(pred_fake_A))

            # Identity loss
            idt_B = g_model_AtoB(real_B)
            idt_A = g_model_BtoA(real_A)
            loss_idt_B = identity_loss(idt_B, real_B)
            loss_idt_A = identity_loss(idt_A, real_A)

            # Cycle-consistency loss
            cycle_B = g_model_AtoB(fake_A_tensor)
            cycle_A = g_model_BtoA(fake_B_tensor)
            loss_cycle_B = cycle_loss(cycle_B, real_B)
            loss_cycle_A = cycle_loss(cycle_A, real_A)

            # Total loss
            loss_g_BtoA = loss_adv_BtoA + 5 * loss_idt_B + 10 * loss_cycle_B
            loss_g_AtoB = loss_adv_AtoB + 5 * loss_idt_A + 10 * loss_cycle_A

            # Backpropagation and optimization
            loss_g_BtoA.backward()
            optimizer_g_BtoA.step()
            optimizer_c_BtoA.step()

            ##############################
            # Update Discriminators
            ##############################

            # Set discriminators' gradients to zero
            optimizer_d_A.zero_grad()
            optimizer_d_B.zero_grad()

            # Real loss
            pred_real_A = d_model_A(real_A)
            pred_real_B = d_model_B(real_B)
            loss_real_A = adversarial_loss(pred_real_A, torch.ones_like(pred_real_A))
            loss_real_B = adversarial_loss(pred_real_B, torch.ones_like(pred_real_B))

            # Fake loss
            pred_fake_A = d_model_A(fake_A_tensor.detach())
            pred_fake_B = d_model_B(fake_B_tensor.detach())
            loss_fake_A = adversarial_loss(pred_fake_A, torch.zeros_like(pred_fake_A))
            loss_fake_B = adversarial_loss(pred_fake_B, torch.zeros_like(pred_fake_B))

            # Total loss
            loss_d_A = (loss_real_A + loss_fake_A) * 0.5
            loss_d_B = (loss_real_B + loss_fake_B) * 0.5

            # Backpropagation and optimization
            loss_d_A.backward()
            optimizer_d_A.step()

            loss_d_B.backward()
            optimizer_d_B.step()

            ##############################
            # Summarize Performance
            ##############################


        save_models(epoch, g_model_AtoB, g_model_BtoA)   
        print(f"Epoch [{epoch+1}/{n_epochs}] | Generator Loss: {loss_g_BtoA} | Discriminator Loss: {loss_g_AtoB}")
   

            


# In[ ]:


for batch in train_dataloader:
    source, target = batch
    print(f'Source batch shape: {source.shape}')
    print(f'Target batch shape: {target.shape}\n')
    break


# In[ ]:


image_shape = source.shape[1:]


# In[ ]:


# generator: A -> B
g_model_AtoB = define_generator(image_shape).to(device)
# generator: B -> A
g_model_BtoA = define_generator(image_shape).to(device)
# discriminator: A -> [real/fake]
d_model_A = define_discriminator(image_shape).to(device)
# discriminator: B -> [real/fake]
d_model_B = define_discriminator(image_shape).to(device)
# composite: A -> B -> [real/fake, A]
c_model_AtoB, optimizer_c_AtoB  = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA,optimizer_c_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)


# In[ ]:


from datetime import datetime 
start1 = datetime.now() 
# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, train_dataloader, epochs=500)

stop1 = datetime.now()
#Execution time of the model 
execution_time = stop1-start1
print("Execution time is: ", execution_time)


# In[ ]:





# In[ ]:





# In[ ]:




