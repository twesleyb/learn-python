def repeat(fun):
    ''' A decorator to repeat a function. ''' 
    def wrapper():
        fun()
        fun()
    return wrapper
# EOF
