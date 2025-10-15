
import os
import numpy as np
import matplotlib.pyplot as plt
import copy

sets=[]
for ne,e in enumerate(err):
    if 'Power' in e:
        sets.append( set( list(np.where(np.array(e['Power'])>2)[0])))

for ne,e in enumerate(sets):
    if ne==0:
        theset = copy.copy(e)
    else:
        theset = e.intersection(theset)

fig,ax=plt.subplots(1,2)
for ne,e in enumerate(err):
    if 'Power' in e:
        #sets.append( set( list(np.where(np.array(e['Power'])>2)[0])))
        ax[0].hist( e['Power'], histtype='step', bins=100)
        x=sorted(e['Power'])[:-150]
        y = np.arange(len(x))/len(x)
        ax[1].plot(x,y)
fig.savefig('%s/plots/all.pdf'%os.environ['HOME'])

