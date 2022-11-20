import codecs
file = open("/home/nbm/PycharmProjects/lstm_text.txt", "rb").read().decode('utf-8').split("\n")
k=list(file)
class textgen(object):
    def __init__(self):
         self.lineno = len(file)-1
         self.rt = "/home/nbm/PycharmProjects/lstm_text/"
         self.lin=0

    def input(self):
        while self.lin < self.lineno:
            self.fname =self.rt+ "/01%04d"%self.lin
            self.lin += 1
            return self.fname
            #self.out = open(self.fname + ".gt.txt", "w")
fun = textgen()
for i in range(len(k)-1):
    if len(k[i])<=0: continue
    with codecs.open(fun.input() + ".gt.txt", "w", 'utf-8') as stream:
        stream.write(k[i] + "\n")




