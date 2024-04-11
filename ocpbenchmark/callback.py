import crocoddyl

class benchmark_callback(crocoddyl.CallbackAbstract):
    def __init__(self, attributes_name):
        super().__init__()
        self.KKTs = [] #KKT conditions
        self.grads = [] # gradients   
        self.ffess = [] # dynamics infeasibilities
        self.gfess = [] # ineqaulity infeasibilities
        self.hfess = [] # equality infeasibilities
        self.merits = [] # merit function values
        self.steps = [] # step lengths
    def __call__(self, solver):
        self.problem.callback(solver)