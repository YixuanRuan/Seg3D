class Format:
    """ Format util class

    """
    @staticmethod
    def head(str):
        print('############### %s ###############' % str)

    @staticmethod
    def tail():
        print('#############################################')

    @staticmethod
    def tick(elapsed):
        print("Time used: %.3fs" % elapsed)


    @staticmethod
    def utilsHead(str):
        print('--------------- %s ---------------' % str)

    @staticmethod
    def utilsTail():
        print('---------------------------------------------')