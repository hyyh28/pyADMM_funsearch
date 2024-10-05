def evolve(func):
    def wrapper(*args, **kwargs):
        print(f"Running evolve on {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

def run(func):
    def wrapper(*args, **kwargs):
        print(f"Running {func.__name__}")
        return func(*args, **kwargs)
    return wrapper
