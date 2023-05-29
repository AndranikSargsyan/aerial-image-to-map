#!/usr/bin/env python
# coding: utf-8

# In[3]:


import random
import numpy as np
from numpy.random import randint
#from numpy import asarray


# In[ ]:


def save_models(step, g_model_AtoB, g_model_BtoA):
    # Save the first generator model
    filename1 = 'g_model_AtoB_%06d.pth' % (step+1)
    torch.save(g_model_AtoB.state_dict(), filename1)
    # Save the second generator model
    filename2 = 'g_model_BtoA_%06d.pth' % (step+1)
    torch.save(g_model_BtoA.state_dict(), filename2)
    print('> Saved: %s and %s' % (filename1, filename2))


# In[ ]:


def update_image_pool(pool, images, max_size=50):
    selected = []
    for image in images:
        if len(pool) < max_size:
            # Stock the pool
            pool.append(image)
            selected.append(image)
        elif random.random() < 0.5:
            # Use image, but don't add it to the pool
            selected.append(image)
        else:
            # Replace an existing image and use replaced image
            ix = random.randint(0, len(pool) - 1)
            selected.append(pool[ix])
            pool[ix] = image
    return np.asarray(selected)


# In[ ]:




