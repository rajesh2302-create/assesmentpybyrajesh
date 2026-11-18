class insufficientfunderror(Exception):
    pass
class account:
 def __init__(self,name,balance):
  self.name=name
  self.balance=balance

 def withdraw(self,amount):
   if amount>self.balance:
     raise insufficientfunderror(f"insufficient fund balance{self.balance}requested amount{amount}")
   self.balance-=amount
   print(f"withdraw amount{amount}balance{self.balance}")

class savingaccount(account):
  def __init__(self,name,balance,min_balance=500):
    super().__init__(name,balance)
    self.min_balance=min_balance
  def withdraw(self,amount):
    if self.balance-amount<self.min_balance:
      raise insufficientfunderror(f"insufficientbalance{self.balance}requestedamount{amount}")
    self.balance-=amount
    print(f"{self.name}withdraw{amount}balance{self.balance}")  
priya=savingaccount("priya",1500)
try:
  priya.withdraw(1200)
except insufficientfunderror as e:
  print(f"error:{e}")
try:
   priya.withdraw(800)
except insufficientfunderror as e:
   print(f"error:{e}")
print(isinstance(priya,account))       

   
