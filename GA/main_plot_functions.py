from GA.fitness_func import (
    AckleyFunction,
    Deb2Function,
    Deb4Function,
    RastriginFunction,
)

if __name__ == "__main__":
    for func in [
        AckleyFunction,
        RastriginFunction,
        Deb2Function,
        Deb4Function,
    ]:
        func(1).plot()
