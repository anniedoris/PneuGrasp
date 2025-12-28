import nevergrad as ng

class CMA:
    def __init__(self, budget, num_workers, instrum):
        self.budget = budget
        self.num_workers = num_workers
        self.instrum = instrum
        self.optimizer= ng.optimizers.registry['CMA'](parametrization=instrum, budget=self.budget, num_workers=self.num_workers)

    def get_trial_parameters(self):
        return self.optimizer.ask()

    def get_loss(self, x, score):
        self.optimizer.tell(x, score)

    def report_optimal(self):
        self.recommendation = self.optimizer.provide_recommendation().value

        # recommendation = optimizer.minimize(square)  # best value


