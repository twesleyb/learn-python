# Using the get() method with python dictionaries.
users = {382 : "Alice", 590 : "Bob", 951 : "Dilbert"}

def greeting(userid):
    name = users.get(userid, "there") # Default is userid is not found = "Hi there!"
    print(f"Hi {name}!")

greeting(382)

greeting(3333)
