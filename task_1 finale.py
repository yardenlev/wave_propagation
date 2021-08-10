'''
Author: Yarden Levenberg
Email: Yarden.lev@gmail.com
-------------------------------------------------------
Short Description:
This is a solution for the wave equation in acoustic model using explict method
on a XZ-plane where x is surface distance and z is depth [m]
-------------------------------------------------------
'''
import numpy as np
import matplotlib.pyplot as plt


# physical consts
x_source = 3000  # [m]
z_source = 2800  # [m]
dx = dz = 10  # [m]
dt = 0.001  # [s]
end_t = 1
x_layers_separation = [0, 1000, 2600, 4600, 6000]  # x layer's separating points
z_layers_separation = [2600, 4000, 3200, 3600, 2400]  # z layer's separating points
v_layer_1 = 2000  # [m/s]
v_layer_2 = 3000  # [m/s]


# boundary
x_range = [0, 6000]  # [m]
z_range = [0, 6000]  # [m]


# other variables
map_color = 'seismic'  # c-map colors min(blue-white-red)max


# analytic functions:
def F(t):  # source function
    if t > 0.05: return 0
    return t * np.sin(2 * np.pi * t) * np.exp(2 * np.pi * t)


# return the value of poly in given x1
def poly_value(poly, x):
    poly_power = len(poly) - 1
    poly_value = 0
    for i in range(poly_power+1):
        poly_value += (poly[i]*(x**i))
    return poly_value


# add a matrix to another matrix, value by value
def add_matrix(mat1, mat2):
    ret_mat = []
    for i in range(len(mat1)):
        temp = []
        for j in range(len(mat2[i])):
            temp.append(mat1[i][j]+mat2[i][j])
        ret_mat.append(temp)
    return ret_mat


# find and return the spline poly's variables used for calculating the mekadmim
def find_h_b_u_v(x, y):
    h = []
    b = []
    for i in range(0, len(x)-1):
        h.append(x[i+1] - x[i])
        b.append((y[i+1] - y[i]) / h[i])
    u = []
    v = []
    for i in range(1, len(h)):
        u.append(2 * (h[i-1] + h[i]))
        v.append(6 * (b[i] - b[i-1]))
    return h, b, u, v


# find and return the spline poly's mekadmim
def find_a_b_c_d(x, y, h, m):
    a = []
    b = []
    c = []
    d = []
    for i in range(len(h)):
        a.append((m[i + 1] - m[i]) / (6*h[i]))
        b.append((m[i] * x[i + 1] - m[i + 1] * x[i]) / (2 * h[i]))
        c.append((m[i + 1] * x[i] ** 2 - m[i] * x[i + 1] ** 2) / (2 * h[i]) + (y[i + 1] - y[i]) / h[i]
                 + (m[i] * h[i] - m[i + 1] * h[i]) / 6)
        d.append((m[i + 1] * h[i] * x[i] - m[i] * h[i] * x[i + 1]) / 6 + (y[i] * x[i + 1] - y[i + 1] * x[i]) / h[i]
                 - (m[i + 1] * x[i] ** 3 - m[i] * x[i + 1] ** 3) / (6 * h[i]))
    return a, b, c, d


# creates a matrix for layers speed in ground using the cubic spline as a buffer
# between the layers
def layer_speed_matrix(cubic_spline):
    c_matrix = np.zeros((int((x_range[1] - x_range[0]) / dx), int((z_range[1] - z_range[0]) / dz)))
    section = [xx/dx for xx in x_layers_separation]
    for i in range(len(cubic_spline)):
        for x in range(int(section[i]), int(section[i+1])):
            layers_depth = int(poly_value(cubic_spline[i], x*dx)/dz)
            for z in range(int(z_range[1]/dz)):
                if z <= layers_depth:
                    c_matrix[x][z] = v_layer_1
                else:
                    c_matrix[x][z] = v_layer_2
    return c_matrix


# add a vector as a column in the right end of a 2D list
def matrix_with_solution(matrix, vector):
    for i in range(len(matrix)):
        if isinstance(matrix[i], list):
            matrix[i].append(vector[i])
        else:
            matrix[i] = [matrix[i], vector[i]]
    return matrix


# does what it says it does, creating tridiagonal matrix
def create_tridiagonal_matrix(h, u):
    mat = [[0]*(len(h)-1) for dummy in range(len(h)-1)]
    for i in range(len(mat)):
        mat[i][i] = u[i]
        if i < len(mat)-1: mat[i][i+1] = mat[i+1][i] = h[i+1]
    return mat


# solving a tridiagonal matrix with a vector V
def solve_tridiagonal_matrix(matrix, v):
    mat = matrix_with_solution(matrix, v)
    for i in range(len(mat)):  # making main diagonal = 1
        mat[i] = [x / mat[i][i] for x in mat[i]]
        if i+1 != len(mat):
            mat[i + 1] = [mat[i + 1][j] - mat[i + 1][i] * mat[i][j] for j in range(len(mat[i]))]
    for i in range(1, len(mat)):  # going backwards making everything 0 but diagonal
        mat[-i - 1] = [mat[-i - 1][j] - mat[- i - 1][-i-1] * mat[-i][j] for j in range(len(mat[i]))]
    return mat


# cubic spline for layers differentiating using the xz_dict
# creating polynomials based on a series of points x, f(x)
# and gives back the cubic spline interpolation
def cubic_spline(x, z):
    h, b, u, v = find_h_b_u_v(x, z)
    mat = create_tridiagonal_matrix(h, u)
    mat = solve_tridiagonal_matrix(mat, v)
    m = []
    for i in range(len(mat)+2):
        if i == 0: m.append(0)
        elif i != len(mat)+1: m.append(mat[i-1][-1])
        else: m.append(0)
    a, b, c, d = find_a_b_c_d(x, z, h, m)
    return [[d[i], c[i], b[i], a[i]] for i in range(len(a))]


# given two matrices of the wave function in current phase and previous phase, using
# finite differences 4th order on space and finite differences 2nd order on time
# to create the next phase of the 2D wave equation
def wave_2D(matrix_now, matrix_before, t, c_matrix):
    next_matrix = np.zeros((int((x_range[1] - x_range[0]) / dx), int((z_range[1] - z_range[0]) / dz)))
    for i in range(2, int(x_range[1] / dx) - 2):
        for j in range(2, int(z_range[1] / dz) - 2):
            next_matrix[i][j] = (c_matrix[i][j]**2 * dt**2) / (12 * dx ** 2) * \
                    (16 * (matrix_now[i + 1][j] + matrix_now[i - 1][j]) - matrix_now[i + 2][j] -
                     matrix_now[i - 2][j] + 16 * (matrix_now[i][j + 1] + matrix_now[i][j - 1]) -
                     matrix_now[i][j + 2] - matrix_now[i][j - 2] - 60 * matrix_now[i][j]) + 2 *\
                    matrix_now[i][j] - matrix_before[i][j]
            if i == int(x_source/dx) and j == int(z_source/dz):
                next_matrix[i][j] += F(t) * (dt ** 2)
    return next_matrix, matrix_now


# plot poly on XZ plane
def plot_function(poly, area, color):
    ax = plt.gca()
    xi = np.linspace(area[0], area[1], area[1])
    yi = poly_value(poly, xi)
    ax.plot(xi, yi, color=color, linewidth=1)


# formatting the plot and it boundaries for the model in the question
def plot_formatting():
    plt.rcParams["figure.figsize"] = (7, 7)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.grid(b=True, which='major', color='k', linestyle='--')
    ax.xaxis.tick_top()
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    plt.title("Wave equation in acoustic model using explict method")
    ax.set_xlim((x_range[0], x_range[1]))
    ax.set_ylim((z_range[1], z_range[0]))
    plt.text(x_range[1] - 1500, 500, "V ="+str(v_layer_1)+" [m/s]")
    plt.text(x_range[1] - 1500, z_range[1] - 500, "V =" + str(v_layer_2) + " [m/s]")


# this is main function
def main():
    matrix_now = np.zeros((int((x_range[1] - x_range[0]) / dx), int((z_range[1] - z_range[0]) / dz)))
    matrix_before = matrix_now
    cubic_s = cubic_spline(x_layers_separation, z_layers_separation)
    c_matrix = layer_speed_matrix(cubic_s)
    time_series = np.around(np.arange(0, end_t + dt, dt), decimals=7)
    counter = 0
    for t in time_series:
        print("t = ", t, " [s]")
        matrix_now, matrix_before = wave_2D(matrix_now, matrix_before, t, c_matrix)
        if t == 0.4 or t == 0.15 or t == 0.7 or t == 1:
        # if t < end_t:
            plot_formatting()
            plt.scatter(x_source, z_source, marker="*", alpha=0.4, s=150, color='y', edgecolor='black')
            plt.colorbar(plt.imshow(matrix_now.T, cmap=map_color, alpha=0.9, vmin=-1E-7, vmax=1E-7,
                       extent=(x_range[0], x_range[1], z_range[0], z_range[1]), origin='lower'))
            for i in range(len(cubic_s)):
                plot_function(cubic_s[i], [x_layers_separation[i], x_layers_separation[i + 1]], 'k')
            plt.text(x_range[0] + 50, z_range[1] - 50, "t =" + str(t) + " [s]")
            # if counter % 10 == 0:
                # plt.savefig('wave2D_i='+str(counter/1000)+".png")
            plt.show()
            # plt.close()
        counter+=1


main()


