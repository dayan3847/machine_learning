import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == '__main__':
    _file_name = 'data_points.csv'

    # load data points
    data_points: np.array = np.loadtxt(_file_name, delimiter=',').T
    # Data
    plt.title('Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axvline(color='black')
    plt.axhline(color='black')
    # points
    plt.scatter(data_points[0], data_points[1], color='gray', label='data points')
    plt.legend()
    plt.grid()
    plt.show()

    print('\033[92m' + 'loading data... ' + '\033[0m')
    history_df: pd.DataFrame = pd.read_csv('history.csv')

    plt.title('Error')
    plt.xlabel('iteration')
    plt.ylabel('error')
    plt.axvline(color='black')
    plt.axhline(color='black')
    plt.plot(history_df['error'], label='error', color='red')
    plt.legend()
    plt.grid()
    plt.show()

    xf = np.arange(0, 1, 0.01)
    history_polynomial = history_df['polynomial'].tolist()
    yf_initial = [sp.sympify(history_polynomial[0]).subs('x', x) for x in xf]
    yf_final = [sp.sympify(history_polynomial[-1]).subs('x', x) for x in xf]
    plt.title('Data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axvline(color='black')
    plt.axhline(color='black')
    # points
    plt.scatter(data_points[0], data_points[1], color='gray', label='data points')
    # polynomial initial
    plt.plot(xf, yf_initial, label='polynomial initial', color='orange')
    # polynomial final
    plt.plot(xf, yf_final, label='polynomial final', color='green')
    plt.legend()
    plt.grid()
    plt.show()

    total_frames = 1001


    def update_plot(frame):
        print('\033[92m' + f'frame: {frame} de {total_frames}' + '\033[0m')
        ff = frame * 4
        plt.cla()
        yf_current = [sp.sympify(history_polynomial[ff]).subs('x', x) for x in xf]
        yf_best = [sp.sympify(history_polynomial[ff]).subs('x', x) for x in xf]

        plt.title("Iteration: {}".format(ff))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axvline(color='black')
        plt.axhline(color='black')
        # points
        plt.scatter(data_points[0], data_points[1], color='gray', label='data points')
        # polynomial initial
        plt.plot(xf, yf_initial, label='polynomial initial', color='orange')
        # polynomial current (discontinuous)
        plt.plot(xf, yf_current, label='polynomial current', color='blue', linestyle='dashed')
        # polynomial best
        plt.plot(xf, yf_best, label='polynomial best', color='green')
        plt.legend()
        plt.grid()
        return plt.gcf()


    print('\033[92m' + 'creating animation... ' + '\033[0m')
    fig = plt.figure()
    ani = animation.FuncAnimation(fig, update_plot, frames=1001, interval=200)
    ani.save('mi_animacion.mp4', writer='ffmpeg')
