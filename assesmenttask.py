import io
import csv
def calculate_bonus(salary,rate=0.1,*extras,**options):
    bonus=(salary*rate)+sum(extras)
    if 'cap'in options:
        bonus=min(bonus,options['cap'])
    return(bonus)
def log_call(func):
    def wrapper(*args,**kwargs):
        print(f"calling{func.__name__}")
        return func(*args,**kwargs)
    return wrapper
#base class employee
class Employee:
    def __init__(self,name,salary):
        self.name=name
        self.salary=salary
    @property
    def annual_salary(self):
        return self.salary*12
    def get_details(self):
        return f"{self.name}-{self.salary}"
    #derived class manager with decorator
class manager(Employee):
        def __init__(self,name,salary,team_size):
            super().__init__(name,salary)
            self.team_size=team_size
        @log_call
        def get_details(self):
            return f"{super().get_details()}-teamsize:{self.team_size}"
#data processing
csv_data="""name,department,salary
rajesh,it,30000
vennila,hr,25000
sudhan,it,50000
vimal,sales,60000
mugesh,finance,75000"""
#control flow
reader=csv.DictReader(io.StringIO(csv_data.strip()))
for row in reader:
    name=row['name']
    salary=int(row['salary'])

    if salary < 50000:
        category="low salary"
    elif salary < 70000:
        category="medium salary"
    else:
        category="high salary"

    print(f"{name}-{category}")
    #manager details for output verificaton
mgr=manager("sudhan",75000,5)
print(mgr.get_details())


