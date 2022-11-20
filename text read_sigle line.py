import codecs
file = open("/home/nbm/printed_page.txt", "rb").read().decode('utf-8').split("\n")
k=list(file)
rt =open("/home/nbm/kk.txt",'w')
for i in range(len(k)-1):
    if len(k[i])<=0: continue
    rt.write(k[i].encode('utf-8') + "")
rt.close()
e=[]
for line in k:
     #if line==" ": continue
     e.append(line.encode('utf-8'))
print e