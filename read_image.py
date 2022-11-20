import codecs
import glob
import PIL
from scipy.misc import imsave
path1 = '/home/nbm/birh_im//'
imagepath = sorted(glob.glob(path1 + '/*.bin.png'))
print imagepath[19]
class textgen(object):
    def __init__(self):
         self.lineno = len(imagepath)-1
         self.rt = "/home/nbm/bire_image/"
         self.lin=0

    def input(self):
        while self.lin < self.lineno:
            self.fname =self.rt + "/01%06d"%self.lin
            #imsave(self.fname + ".bin.png", imagepath[self.lin])
            self.lin += 1
            return self.fname

             #self.out = open(self.fname + ".gt.txt", "w")
fun = textgen()
for i in range(5):
    imsave(fun.input()+".bin.png",imagepath[i])




