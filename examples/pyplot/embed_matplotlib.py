"""Include background images in the rendering scene
(e.g. generated by matplotlib)"""
import matplotlib.pyplot as plt
from vedo import *

tmsh = TetMesh(dataurl+"limb.vtu")
msh = tmsh.tomesh().shrink(0.8)

# Create a histogram with matplotlib
fig = plt.figure()
plt.hist(msh.celldata["chem_0"], log=True)
plt.title(r'$\mathrm{Matplotlib\ Histogram\ of\ log(chem_0)}$')

pic1 = Image(fig).clone2d("bottom-right", scale=0.5).alpha(0.8)
pic2 = Image(dataurl+"images/embryo.jpg").clone2d('top-right')

show(msh, pic1, pic2, __doc__, bg='lightgrey', axes=1)
