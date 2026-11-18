def add(a,b):
    return a+b
def average(numbers):
    if not numbers:
        return 0
    return sum(numbers)/len(numbers)
if __name__=="main":
    print("testing math_tools.py directly:")
    print("add(5+3):",add(5,3))
    print("average[10,20,30]:",average([10,20,30]))