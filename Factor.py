class Factor:
    # variable index
    variable: int
    # variable degree
    degree: int

    # init
    def __init__(self, variable: int = 0, degree: int = 0):
        self.variable = variable
        self.degree = degree
