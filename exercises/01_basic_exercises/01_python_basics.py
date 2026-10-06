B=2
T=3
C=4

a_list=[B,T,C]

for i in a_list:
    print(i)

def multiply(x,y):
    return x*y

class ModelConfig:
    def __init__(self,number):
        self.number = number

    def present(self):
        print(self.number)

config = ModelConfig(C)
config.present()