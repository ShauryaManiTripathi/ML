def openfile(name):
    f = open(name, 'w')
    f.close()
    print("File created successfully")
    return

import random
def fillfile(name,count):
    f = open(name, 'w')
    f.write('x' + "," + 'y' + "\n")
    for i in range(count):
        f.write(str(random.randint(1,100)) + "," + str(random.randint(1,100)) + "\n")
    f.close()
    print("File filled successfully")
    return

openfile("data.csv")
fillfile("data.csv",30)



import csv
def readfile(name):
    f = open(name, 'r')
    reader = csv.reader(f)
    count = 0
    for row in reader:
        if count > 5:
            break
        print(row)
        count += 1
    f.close()
    return

readfile("data.csv")


# find maximum and minimum of each feature

def findmaxmin(name):
    f = open(name, 'r')
    reader = csv.reader(f)
    count = 0
    x = []
    y = []
    for row in reader:
        if count == 0:
            count += 1
            continue
        x.append(int(row[0]))
        y.append(int(row[1]))
    f.close()
    print("Max x:",max(x))
    print("Min x:",min(x))
    print("Max y:",max(y))
    print("Min y:",min(y))
    return

findmaxmin("data.csv")
