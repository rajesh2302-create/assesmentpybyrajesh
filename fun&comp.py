prices = [500,1200,800,2500,300]

def apply_discount(price,discount=10):
 return float(price*(1-discount/100))
def total(*args):
  return sum(args)
def describe(**kwargs):
  for key,value in kwargs.items():
    print(f"{key}:{value}")

print("--step1:apply_discount testing")
print(f"apply_discount(1000)---{apply_discount(1000)}")
print(f"apply_discount(1000,25)---{apply_discount(1000,25)}")
print("step2:total and describe testing")
print(f"total(500,100,200)---{total(500,100,200)}")
describe(name="rajesh",dept="gen ai")
print("list comprehsion")
above_1000=[p for p in prices if p>1000]
print(f"above_1000--{above_1000}")
print("map with lambda")
with_gst=list(map(lambda p: round(p*1.18,2),prices))
print(f"with_gst--{with_gst}")
print("filter with lambda")
below_1000=list(filter(lambda p:p<1000,prices))
print(f"below_1000---{below_1000}")

