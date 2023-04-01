#!/usr/bin/env python
# coding: utf-8

# In[2]:


def SaveWthAxis(InputImageTest):

    plt.axis('off')

    spec = plt.imshow(InputImageTest)

    plt.savefig('spec',bbox_inches='tight',transparent=True, pad_inches=0)


# In[ ]:




