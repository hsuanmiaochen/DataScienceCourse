# you must use python 3.10
# For linux, you must use download HomeworkFramework.cpython-310-x86_64-linux-gnu.so
# For Mac, you must use download HomeworkFramework.cpython-310-darwin.so
# If above can not work, you can use colab and download HomeworkFramework.cpython-310-x86_64-linux-gnu.so and don't forget to modify output's name.

import numpy as np
from HomeworkFramework import Function
import nevergrad as ng


class Nevergradoptimizer(Function): # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func) # must have this init to work normally

        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func

        self.eval_times = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)



    def get_optimal(self):
        return self.optimal_solution, self.optimal_value
    
    def objective(self, input_parameters): #
        value = self.f.evaluate(self.target_func, input_parameters)
        self.eval_times += 1

        # Check if the returned value is a float
        try:
            value = float(value)
        except ValueError:
            print(f"Error in evaluation: {value}")
            return float('inf')  

        if value < self.optimal_value:
            self.optimal_solution[:] = input_parameters
            self.optimal_value = value

        return value

    def run(self, FES):
        optimizer = ng.optimizers.NGO(parametrization=self.dim, budget=FES)
        recommendation = optimizer.minimize(self.objective)
        print("optimal: {}\n".format(self.get_optimal()[1]))
        return self.get_optimal()

            


if __name__ == '__main__':
    func_num = 1
    fes = 0
    #function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000
        else:
            fes = 2500

        # you should implement your optimizer
        op = Nevergradoptimizer(func_num)
        op.run(fes)

        best_input, best_value = op.get_optimal()
        print(best_input, best_value)

        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format('112138502'.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1
