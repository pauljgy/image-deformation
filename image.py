import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.widgets import Slider

source = Image.open("cavs.webp").convert('L')
source = np.asarray(source.resize((128, 128)))

target = Image.open("lakers.webp").convert('L')
target = np.asarray(target.resize((128, 128)))

def deform(source, target, t):
    image = np.add(np.multiply(1-t, source), np.multiply(t, target))
    return image

temp = plt.imshow(deform(source, target, 0.2))

axb = plt.axes([0.18, 0.01, 0.65, 0.05])
sb = Slider(ax=axb, label="t", valmin=0, valmax=1, valinit=0)

def update(val):
    t = sb.val
    plt.clf()
    plt.imshow(deform(source, target, t))

sb.on_changed(update)

plt.show()