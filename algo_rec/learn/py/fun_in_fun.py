def funouter(a=1,b=2):
    def funinnter():
        print(a)
    funinnter()


funouter(1,2)