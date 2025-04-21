from GA.fitness_func import RastriginFunction, Deb4Function


if __name__ == "__main__":
    for func in [RastriginFunction, Deb4Function]:
        func(1).plot()
