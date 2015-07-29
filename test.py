b = 1
# def __init__(self,b):
#   self.b = b
def mod_b(b,num):
	b += 2
def mod_other(b):
	mod_b(b,4)

if __name__ == '__main__':
    b = 1
    mod_other(b)
    print(b)