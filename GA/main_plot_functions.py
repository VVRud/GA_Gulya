from GA.fitness_func import Deb4Function, RastriginFunction

if __name__ == "__main__":
    for func in [RastriginFunction, Deb4Function]:
        func(1).plot()
