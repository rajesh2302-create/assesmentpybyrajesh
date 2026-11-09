import csv
import json
names=["rajesh","suthan","vennila","ajesh","vimal"]
with open ("names.txt",'w')as f:
    for name in names:
     f.write(name + "\n")


with open ("names.txt",'r')as f:
    for i, line in enumerate(f,1):
       print(f"{i}. {line.strip()}")
students_data = [{"name":"rajesh","marks":"90"},
               {"name":"suthan","marks":"40"},
               {"name":"vennila","marks":"80"},
               {"name":"ajesh","marks":"50"},
               {"name":"vimal","marks":"30"}]      

with open("students.csv","w", newline="")as f:
          writer = csv.DictWriter(f,fieldnames=["name","marks"])
          writer.writeheader()
          writer.writerows(students_data)
above_60=[]
with open("students.csv","r")as f:
       reader=csv.DictReader(f)             
       for row in reader:
            if int(row["marks"])>60:
              above_60.append(f"{row['name']}{row['marks']}")
print("above 60:" + " | ".join(above_60))
with open("students.json","w") as f:
    json.dump(students_data,f,indent=2)

with open("students.json","r") as f:
   loaded_json=json.load(f)
try:
   with open("missing.txt","r")as f:
       content=f.read()
except FileNotFoundError:
   print("missing.txt->file not found")          