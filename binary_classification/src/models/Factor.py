class Factor:
    # variable index
    variable: int
    # variable degree
    degree: int

    # init
    def __init__(self, variable: int = 0, degree: int = 0):
        self.variable = variable
        self.degree = degree

    def __str__(self) -> str:
        if 0 == self.degree:
            return '1'
        result = f"x{self.variable}"
        if 1 != self.degree:
            result += f"^{self.degree}"
        return result
