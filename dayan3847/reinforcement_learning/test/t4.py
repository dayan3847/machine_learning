# mostrar un círculo con openCV

import cv2
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt

def asd():
    # Crear una imagen negra
    img = np.zeros((512, 512, 3), np.uint8)

    # Dibujar un círculo azul en el centro de la imagen
    cv2.circle(img, (256, 256), 63, (255, 0, 0), -1)

    # Mostrar la imagen

    cv2.imshow('image', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('click on points')
#
# line, = ax.plot(np.random.rand(100), 'o', picker=5)  # 5 points tolerance
#
# plt.show()
#

def qe():
    fig2 = plt.figure()

    x = np.arange(-9, 10)
    y = np.arange(-9, 10).reshape(-1, 1)
    base = np.hypot(x, y)
    ims = []
    for add in np.arange(15):
        ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

    im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
                                       blit=True)
    # To save this second animation with some metadata, use the following command:
    # im_ani.save('im.mp4', metadata={'artist':'Guido'})

    plt.show()


from threading import Thread

def run():
    Thread(target=qe).start()
    # Thread(target=asd).start()
    asd()

run()