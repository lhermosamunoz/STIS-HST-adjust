'''
Script to select all the contributions of the broad H alpha component to the global Halpha+NII complex
in comparison to the values from C15
'''
import numpy as np
import matplotlib.pyplot as plt

NGC3245_C15 = 79
NGC4594_C15 = 77
NGC4736_C15 = 93

NGC3245_HM19 = 46.7
NGC3245_2_HM19 = 25.74
NGC4374_HM19 = 65.31
NGC4552_HM19 = 70.47
NGC4594_HM19 = 15.95
NGC4676_HM19 = 88.68
NGC4698_HM19 = 38.53
NGC4736_HM19 = 82.03

plt.ion()
plt.show()
plt.figure(figsize=(8,7))
plt.plot(NGC3245_HM19,NGC3245_C15,marker='o',linestyle='None',label='NGC3245')
plt.plot(NGC4594_HM19,NGC4594_C15,marker='s',linestyle='None',label='NGC4594')
plt.plot(NGC4736_HM19,NGC4736_C15,marker='o',linestyle='None'label='NGC4736')
plt.plot(np.arange(20,101,1),np.arange(20,101,1),'k--')
plt.xlabel('f$_{Halpha}$ this work',fontsize=18)
plt.ylabel('f$_{Halpha}$ C15',fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(50,100)
plt.xlim(xmax=100)
plt.legend(loc='best',fontsize=13)
