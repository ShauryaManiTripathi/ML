x=10
y=20

print(f"before swap x={x}, y={y}")

x,y=y,x

print(f"after swap x={x}, y={y}")

temp=x
x=y
y=temp

print(f"after another swap using temp x={x}, y={y}")