class insufficientfunderror(Exception):
 pass
    
class Account:
  def __init__(self,owner,balance=0):
    self.owner=owner
    self.balance=balance
  def deposit(self,amount):
    self.balance+=amount
    print(f"{self.owner}deposited{amount} balance{self.balance}") 
  def withdraw(self,amount):
   if amount > self.balance:
     raise insufficientfunderror(f"insufficient fund:balance{self.balance}requested amount{amount}") 
   self.balance-=amount      
   print(f"{self.owner}withdrew{amount}balance{self.balance}")
  def show(self):
   print(f"account owner:{self.owner},balance:{self.balance}")
try:
           acct = Account ("rajesh",1000)
           acct.deposit(500)
           acct.withdraw(2000)
except insufficientfunderror as e:
    print(f"error:{e}")
finally:
    print("transaction finished")

           

     
