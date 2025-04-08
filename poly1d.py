import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

cars_age= [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
speed = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
#cars_age = [5,7,8,7,2,17,2,9,4,11,12,9,6]
#speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
mymodel=numpy.poly1d(numpy.polyfit(cars_age,speed,3))
myline=numpy.linspace(1,22,100)
plt.scatter(cars_age,speed)
plt.plot(myline,mymodel(myline))
plt.show()
car_speed=mymodel(16)
#print(r2_score(y,mymodel(x)))
print(r2_score(speed,mymodel(cars_age)))
print("16 year old car speed=",car_speed)
