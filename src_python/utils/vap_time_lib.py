import time
def vap_time_decorator(func):
    def inner1(*args, **kwargs):
        t1 = time.time()
        func(*args, **kwargs)
        print("Total time taken in : ", func.__name__, time.time()-t1)
    return inner1