from random import random, randint
import numpy as np
from math import exp, sqrt
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Optional, Union
from matplotlib import cm

# Abstract class for annealing
class AbstractAnnealModel:

    # Get the "badness" function
    def evaluate(self):
        return 0

    # Make a change that can be reverted
    def revertable_change(self, args = None):
        return

    # Abort the last change made. Note it can work only once in a row.
    # Often it can be done more optimally than just making class copy
    def abort_last_change(self):
        return

    def get_params(self):
        return

class FunctionAnnealModel(AbstractAnnealModel):

    def __init__(self, function, start_args):
        self.function = function
        self.args = start_args
        self.prev_args = None

    def evaluate(self):
        return self.function(self.args)

    def revertable_change(self, changes):
        self.prev_args = self.args.copy()
        for i in range(len(self.args)):
            self.args[i] = self.args[i] + changes[i]

    def abort_last_change(self):
        if self.prev_args is None:
            raise AssertionError("No change was made to undo")
        self.args = self.prev_args.copy()
        self.prev_args = None

    def get_params(self):
        return self.args


class SudokuAnnealModel(AbstractAnnealModel):

    def __init__(self, board):
        self.sq_size = 3
        self.badness = None
        self.last_change = None

        self.board = []
        self.fixed = []
        self.free_cells = []
        for i in range(self.sq_size ** 2):
            board_line = []
            fixed_line = []
            free_line = []
            for j in range(self.sq_size ** 2):
                cell = board[i][j]
                if (cell == '-'):
                    board_line.append(-1)
                    fixed_line.append(False)
                    free_line.append(j)
                else:
                    cell_integer = int(cell)
                    board_line.append(cell_integer)
                    fixed_line.append(True)
            self.board.append(board_line)
            self.fixed.append(fixed_line)
            self.free_cells.append(free_line)
        self.fill_in()
        self.true_evaluate()

    def fill_in(self):
        for i in range(self.sq_size ** 2):
            line_numbers = set()
            for j in range(self.sq_size ** 2):
                cell = self.board[i][j]
                if cell != -1:
                    line_numbers.add(cell)
            iter = 1
            for j in range(self.sq_size ** 2):
                if (self.board[i][j] == -1):
                    while (iter in line_numbers):
                        iter += 1
                    self.board[i][j] = iter
                    iter += 1

    def true_evaluate(self):
        badness = 0
        for i in range(self.sq_size ** 2):
            badness += self.get_row_badness(i)

        for i in range(self.sq_size):
            for j in range(self.sq_size):
                badness += self.get_square_badness(i, j)
        self.badness = badness

    def evaluate(self):
        return self.badness


    def get_row_badness(self, row_num):
        badness = self.sq_size ** 2
        square_numbers = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(self.sq_size ** 2):
            cell = self.board[i][row_num]
            if square_numbers[cell - 1] == 0:
                badness -= 1
            square_numbers[cell - 1] = 1

        return badness

    def get_square_badness(self, i, j):
        badness = self.sq_size ** 2
        square_numbers = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for k in range(self.sq_size):
            for l in range(self.sq_size):
                cell = self.board[i*self.sq_size + k][j*self.sq_size + l]
                if square_numbers[cell - 1] == 0:
                    badness -= 1
                square_numbers[cell - 1] = 1
        return badness

    def make_change(self, change):
        row_num = change[0]
        first_idx = change[1]
        second_idx = change[2]

        first_row_badness = self.get_row_badness(first_idx)

        second_row_badness = self.get_row_badness(second_idx)

        first_sq_badness = self.get_square_badness(row_num // 3, first_idx // 3)

        second_sq_badness = self.get_square_badness(row_num // 3, second_idx // 3)

        temp = self.board[row_num][first_idx]
        self.board[row_num][first_idx] = self.board[row_num][second_idx]
        self.board[row_num][second_idx] = temp

        new_first_row_badness = self.get_row_badness(first_idx)

        new_second_row_badness = self.get_row_badness(second_idx)

        new_first_sq_badness = self.get_square_badness(row_num // 3, first_idx // 3)

        new_second_sq_badness = self.get_square_badness(row_num // 3, second_idx // 3)

        self.badness += (new_first_row_badness - first_row_badness)
        self.badness += (new_second_row_badness - second_row_badness)
        self.badness += (new_first_sq_badness - first_sq_badness)
        self.badness += (new_second_sq_badness - second_sq_badness)

        self.last_change = change
        return




    def revertable_change(self, args = None):
        row_num = randint(0, self.sq_size ** 2 - 1)
        while (len(self.free_cells[row_num]) < 2):
            row_num = randint(0, self.sq_size ** 2 - 1)

        first_pos = randint(0, len(self.free_cells[row_num]) - 1)
        second_pos = randint(0, len(self.free_cells[row_num]) - 1)
        while (first_pos == second_pos):
            second_pos = randint(0, len(self.free_cells[row_num]) - 1)
        self.make_change([row_num, self.free_cells[row_num][first_pos], self.free_cells[row_num][second_pos]])


    def abort_last_change(self):
        if self.last_change is None:
            raise AssertionError("No change was made to undo")
        self.make_change(self.last_change)
        self.last_change = None

    def get_params(self):
        return self.board

    def get_hash(self):
        hsh = 0
        for i in range(self.sq_size):
            for j in range(self.sq_size):
                cell = self.board[i][j]
                hsh *= 13523
                hsh += cell
                hsh = hsh % 998244353
        return hsh




def exponent_temp_constructor(a, b, skip = 0):
    return lambda i: b * exp(-(i + skip) * a)

def hyperbolic_temp_constructor(a, skip = 0):
    return lambda i: a * (1 / (i + skip))

def rand_n1_1():
    return 2 * random() - 1

def hyperb_change_attrs_constructor(a, skip = 0):
    return lambda i: [a * rand_n1_1() / (i + skip), a * rand_n1_1() / (i + skip)]

def half_hyp_attrs_constructor(a, skip = 0):
    return lambda i: [a * rand_n1_1() / sqrt((i + skip)), a * rand_n1_1() / sqrt((i + skip))]


def const_change_attrs_constructor(a, skip = 0):
    return lambda i: [a * rand_n1_1(), a * rand_n1_1()]

def exp_change_attrs_constructor(a, b, skip = 0):
    return lambda i: [a * rand_n1_1() * exp(-(i + skip) * b), a * rand_n1_1() * exp(-(i + skip) * b)]

def default_tolerance_func(change, temperature):
    return exp(-change / temperature)

def discrete_tolerance_func(change, temperature):
    return temperature



maps = {}
def simulate_annealing(annealing_model : AbstractAnnealModel, temperature_func, change_attrs_func,
                       iterations : int, tolerance_func = discrete_tolerance_func, sudoku = False):

    states = []
    states.append(annealing_model.get_params().copy())
    for i in range(1, iterations + 1):
        if i % 10000 == 0:
            print(str(annealing_model.evaluate()))

        if annealing_model.evaluate() == 0 and sudoku:
            print("Finished successfully, took iterations: " + str(i))
            break

        before_result = annealing_model.evaluate()
        if (sudoku):
            if (not annealing_model.get_hash() in maps):
                maps[annealing_model.get_hash()] = 0
            before_result = before_result + maps[annealing_model.get_hash()] / 10000

        annealing_model.revertable_change(change_attrs_func(i))

        after_result = annealing_model.evaluate()
        if (sudoku):
            if (not annealing_model.get_hash() in maps):
                maps[annealing_model.get_hash()] = 0
            after_result = after_result + maps[annealing_model.get_hash()] / 10000


        if before_result >= after_result:
            states.append(annealing_model.get_params().copy())
            # good change
        elif random() < tolerance_func(after_result - before_result, temperature_func(i)):
            states.append(annealing_model.get_params().copy())
            # bad change, but we're lucky this time
        else:
            annealing_model.abort_last_change()
            # revert bad changes
        if (sudoku):
            maps[annealing_model.get_hash()] += 1

    return states


def example_func(args):
    return 0.1 * (args[0] - args[1] + 1) ** 2 + 3 * (args[0] + args[1] - 2) ** 2


def example_func_2(args, modal_power: float = 2.0):
    modal = modal_power * (np.sin(2 * (args[0] - np.pi / 2)) + np.sin(2 * (args[1] - np.pi / 2)))
    return modal

def example_func_3(args,  noise_power: float = 15.0) -> float:
    noise = noise_power * (np.sin(args[0] * 5) * np.cos(args[1] * 3) * np.sin(np.pi * args[0]) * np.sin(np.pi * args[1] / 2))
    return args[0] ** 2 + 3 * args[1]  ** 2 + noise





class FunctionVisualization:
    """
    Visualization utilities for optimization functions and algorithm trajectories.
    """

    @staticmethod
    def plot_contour_with_trajectory(f: Callable,
                                     points: List[List[float]],
                                     x_range: Tuple[float, float] = (-10, 10),
                                     y_range: Tuple[float, float] = (-10, 10),
                                     levels: int = 10,
                                     title: str = "Annealing Trajectory"):
        """
        Plot contour lines of function with optimization trajectory.

        Args:
            f: Target function f(x, y)
            points: List of points representing the optimization trajectory
            x_range: Range for x-axis
            y_range: Range for y-axis
            levels: Number of contour levels
            title: Plot title
        """
        # Generate grid data
        x = np.arange(x_range[0], x_range[1], 0.05)
        y = np.arange(y_range[0], y_range[1], 0.05)
        xgrid, ygrid = np.meshgrid(x, y)

        # Calculate function values on grid
        z = np.zeros_like(xgrid)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                z[i, j] = f([xgrid[i, j], ygrid[i, j]])

        # Create plot
        plt.figure(figsize=(10, 8))
        cs = plt.contour(xgrid, ygrid, z, levels=levels)
        plt.clabel(cs, inline=True, fontsize=8)

        # Plot trajectory points
        x_points = [p[0] for p in points]
        y_points = [p[1] for p in points]
        plt.plot(x_points, y_points, 'ro-', markersize=4)

        # Mark starting and ending points
        plt.plot(points[0][0], points[0][1], 'go', markersize=6, label='Start')
        plt.plot(points[-1][0], points[-1][1], 'bo', markersize=6, label='End')

        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_3d_surface(f: Callable,
                        x_range: Tuple[float, float] = (-10, 10),
                        y_range: Tuple[float, float] = (-10, 10),
                        title: str = "Function Surface"):
        """
        Plot 3D surface of the function.

        Args:
            f: Target function f(x, y)
            x_range: Range for x-axis
            y_range: Range for y-axis
            title: Plot title
        """
        # Generate grid data
        x = np.arange(x_range[0], x_range[1], 0.2)
        y = np.arange(y_range[0], y_range[1], 0.2)
        xgrid, ygrid = np.meshgrid(x, y)

        # Calculate function values on grid
        z = np.zeros_like(xgrid)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                z[i, j] = f(xgrid[i, j], ygrid[i, j])

        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(xgrid, ygrid, z, cmap=cm.coolwarm, linewidth=0.2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('f(X, Y)')
        ax.set_title(title)

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


easy_sudoku_tests = [
    [
        "4--8-2--7",
        "-18-453--",
        "7---9---4",
        "-31--467-",
        "--9-578-1",
        "82----54-",
        "9---78---",
        "-7--6192-",
        "-85-2-7--"
    ],

    [
        "9--3---71",
        "4378--25-",
        "--5-2--49",
        "-584-9-3-",
        "7--1---98",
        "29--3---4",
        "-8--13---",
        "3-4687---",
        "1--25----"
    ]
]

hard_sudoku_tests = [
    [
        "1----32--",
        "3------51",
        "----1-7--",
        "-1-6-2-94",
        "-9-8----7",
        "-72---1-3",
        "-3---49--",
        "-8----4--",
        "4-9-5--82"
    ],

    [
        "-2918--6-",
        "5-------8",
        "37---2---",
        "--76---31",
        "-----1---",
        "--3827-49",
        "--24-6-5-",
        "--------6",
        "1--7---84"
    ]
]

epic_sudoku_tests = [
    [
        "4---125-3",
        "--8-7----",
        "-------1-",
        "6------9-",
        "-7--2-1-6",
        "-----1-4-",
        "-4-3-----",
        "3----56-2",
        "--------9"
    ],

    [
        "---2----1",
        "-28---9--",
        "----5-3--",
        "4--3---5-",
        "7--------",
        "-1-----47",
        "---574---",
        "9-1--3---",
        "--6--1---"
    ]
]

fun_sudoku_tests = [
    [
        "---------",
        "---------",
        "---------",
        "---------",
        "---------",
        "---------",
        "---------",
        "---------",
        "---------"
    ]
]



def rn_experiment():
    function = FunctionAnnealModel(function=example_func, start_args=[6, 6])
    temperature_func = hyperbolic_temp_constructor(2, skip = 20)
    change_attrs_func = const_change_attrs_constructor(2, skip = 2000)
    iterations = 100
    states = simulate_annealing(function, temperature_func, change_attrs_func, iterations, sudoku=False)

    FunctionVisualization.plot_contour_with_trajectory(example_func, states)
    for i in states:
        print(i)
    return


def sudoku_experiment():
    function = SudokuAnnealModel(board = easy_sudoku_tests[1])
    temperature_func = hyperbolic_temp_constructor(1000, 1000)
    change_attrs_func = const_change_attrs_constructor(1)
    for i in range(100):
        iterations = 1000000
        states = simulate_annealing(function, temperature_func, change_attrs_func, iterations, sudoku=True)
        if function.evaluate() == 0:
            print(i)
            break

    final_board = states[len(states)-1]
    for i in final_board:
        print(i)

    return

if __name__ == '__main__':
    rn_experiment()
