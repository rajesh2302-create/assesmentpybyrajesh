sales=[12000,0,8500,15000,-200,9800]
total=0
for x in sales:
    if x <= 0:
        print(f"skipped invalid value {x}")

        continue
    total+=x
    print(f"added{x}total{total}")
    if total > 30000:
     break

if total > 30000:
        print(f"target reached stopped at {total}")
elif total==30000:
          print(f"exact target reachedstopped at {total}")
else: 
     print("target not reached")  