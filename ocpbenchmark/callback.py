import crocoddyl

class benchmark_callback(crocoddyl.CallbackAbstract):
    def __init__(self, attributes_config):
        super().__init__()
        self.attributes_config = attributes_config
        self.data = {key: [] for key in attributes_config.keys()}

    def __call__(self, solver):
        for attr, solver_attr in self.attributes_config.items():
            # Check if the solver has the attribute. If not, raise an exception.
            if not hasattr(solver, solver_attr):
                raise AttributeError(f"Solver does not have the attribute '{solver_attr}'.")
            # Otherwise, record the attribute's value.
            value = getattr(solver, solver_attr)
            self.data[attr].append(value)