list1=[1,2,3,4]
list2=[5,6,7,8]
list3=list1+list2
print(list3)
evenelement3=list3[1::2]
print(evenelement3)
#using list comprehension
evenelement3=[list3[i] for i in range(1,len(list3),2)]
