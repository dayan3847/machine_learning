class Factor:
    # variable degree
    degree: int
    # variable index
    variable: int

    # init
    def __init__(self, degree: int = 0, variable: int = 0):
        self.degree = degree
        self.variable = variable
