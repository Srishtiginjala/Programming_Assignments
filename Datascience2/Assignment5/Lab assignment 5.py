#Questions 1a) and 1b)

import random 
import matplotlib.pyplot as plt
l=random.choices([0,1],k=10000)
y0=0
y1=0
for x in l:
    if(x==0):
        y=random.choices([0,1],weights=[3,1],k=1)
        if(y[0]==0):
            y0+=1
        else:y1+=1

    if(x==1):
        y=random.choices([0,1],weights=[35,65],k=1)
        if(y[0]==0):
            y0+=1
        else:y1+=1
    
l1=[y0/10000,y1/10000]
plt.bar(["0","1"],l1,width=0.2)
plt.show()
print("The obtained probabilities of Y=0 and y=1 are",l1,"which are appoximately equal to the calculated probabilities of [0.55,0.45]")

#Questions 2a) and 2b)

#here y0x0 means probability that y=0 given that x=0 ie p(Y=0/X=0) and so are remaining
import random 
import matplotlib.pyplot as plt
l=random.choices([0,1],k=10000)
y0x0=0
y1x1=0;y0x1=0;y1x0=0
for x in l:
    if(x==0):
        y=random.choices([0,1],weights=[3,1],k=1)
        if(y[0]==0):
            y0x0+=1
        else:y1x0+=1

    if(x==1):
        y=random.choices([0,1],weights=[35,65],k=1)
        if(y[0]==0):
            y0x1+=1
        else:y1x1+=1
    
l1=[y0x0/10000,y1x0/10000,y0x1/10000,y1x1/10000]
plt.bar(["y0x0","y1x0","y0x1","y1x1"],l1,width=0.2)
plt.show()
print("The obtained probabilities of [y0x0,y1x0,y0x1,y1x1]  are",l1,"which are appoximately equal to the calculated probabilities of [0.375,0.125,0.175,0.325]")
