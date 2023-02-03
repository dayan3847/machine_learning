import matplotlib.pyplot as plt
import numpy as np
import time


f=10
T=1/f

t = np.linspace(0,5*T,300)
diff=t[1]-t[0]

s1=t*1.5
s2=1*np.sin(2*t*np.pi*f)
s3=2*np.exp(-4*t)*np.sin(2*t*np.pi*6*f)

fig, ax = plt.subplots()
#ax.plot(t,s1,t,s2,t,s3)
#plt.show()

if __name__ == '__main__':
    while True:
        t=np.delete(t,0)
        t=np.append(t,t[-1]+diff)
        s1=t*1.5
        s2=1*np.sin(2*t*np.pi*f)
        s3=2*np.exp(-4*t)*np.sin(2*t*np.pi*6*f)
        plt.cla()
        ax.plot(t,s1,'r',t,s2,'b',t,s3,'g')
        plt.pause(0.02)
    plt.show()