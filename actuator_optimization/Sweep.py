

class Sweep:
    def __init__(self, sweep_parameter, min_value, max_value, number):
        self.sweep_parameter = sweep_parameter
        self.min_value = min_value
        self.max_value = max_value
        self.number = number
        self.current_value = min_value

        if number == 1:
            self.step = 1
        else:
            self.step = (max_value - min_value) / (number - 1)


