import numpy as np
import time
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import FormatStrFormatter
from config import *
from math import e
from tkinter import messagebox
from sympy import *
from scipy.optimize import minimize
from scipy.optimize import line_search
from tkinter import *

#settings:
terminate_flag = {}

# For simplicity, current objective function is (x1**2 + 3*x2**2)/2
x1 = Symbol('x1')
x2 = Symbol('x2')
obj = x1**2+x2**2-2*e**(-(((x1-1)**2+x2**2)/0.2))-3 * e**(-((x1 + 1) * (x1 + 1) + x2 * x2) / 0.2)
objective = lambdify((x1, x2), obj, 'numpy')

sample1 = "(x1**2+3*x2**2)/2"
sample2 = 'x1**2+x2**2-2*e**(-(((x1-1)**2+x2**2)/0.2))-3 * e**(-((x1 + 1) * (x1 + 1) + x2 * x2) / 0.2)' #pre:0.01
sample3 = "x1**2+x2**2-e**(x1**2)"

TOTAL_TRACES = {}
CHANGE_FUNC = False

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def f(x):
    global objective
    return objective(x[0],x[1])

def jac_f(x):
    global obj
    dx1 = obj.diff(x1)
    dx2 = obj.diff(x2)
    dx1 = lambdify((x1, x2), dx1, 'numpy')
    dx2 = lambdify((x1, x2), dx2, 'numpy')

    dx = [dx1(x[0],x[1]), dx2(x[0],x[1])]
    return np.array(dx)

def hess_f(x):
    global obj
    dx1x1 = obj.diff(x1).diff(x1)
    dx1x2 = obj.diff(x1).diff(x2)
    dx2x2 = obj.diff(x2).diff(x2)

    dx1x1 = lambdify((x1, x2), dx1x1, 'numpy')
    dx1x2 = lambdify((x1, x2), dx1x2, 'numpy')
    dx2x2 = lambdify((x1, x2), dx2x2, 'numpy')

    dx2 = [[dx1x1(x[0],x[1]), dx1x2(x[0],x[1])],
           [dx1x2(x[0],x[1]), dx2x2(x[0],x[1])]]
    return dx2

def magnitude(v):
    return np.sqrt(v.dot(v))

def exact_line_search(init, iterround=100,error=1e-6,hessian=None):
    x_i, y_i = init[0], init[1]
    traces = []
    res = {'success':False}
    for i in range(1, iterround):
        traces.append(np.array([x_i, y_i]))
        dx_i, dy_i = jac_f(np.asarray([x_i, y_i]))
        step = line_search(f, jac_f,
                                    np.r_[x_i, y_i], -np.r_[dx_i, dy_i],
                                    np.r_[dx_i, dy_i], c2=.05)
        step = step[0]
        if step is None:
            step = 0
        tx, ty = x_i, y_i
        x_i += - step * dx_i
        y_i += - step * dy_i
        if magnitude(np.array([x_i, y_i])-np.array([tx, ty])) < PT_ERROR:
            res['success'] = True
            break
    return Struct(**res), traces




def Newton(init, iterround = 500):
    traces = [init.copy()]
    res = minimize(f, init, method='Newton-CG', jac=jac_f, hess=hess_f, callback=lambda x: traces.append(x), options={'maxiter':iterround})
    return res, traces

def BFGS_method(init, iterround = 500):
    traces = [init.copy()]
    res = minimize(f, init, method='BFGS', jac=jac_f, callback=lambda x: traces.append(x), options={'maxiter': iterround})
    return res, traces

def NelderMead(init, iterround = 500):
    traces = [init.copy()]
    res = {}
    res['success'] = False
    def early_stop(x):

        traces.append(x)
        if magnitude(x - traces[-2]) < PT_ERROR:
            return True

    minimize(f, init, method='Nelder-Mead', callback=early_stop, options={'maxiter': iterround})
    if len(traces) > 2 and magnitude(traces[-2] - traces[-1]) < PT_ERROR:
        res['success'] = True

    return Struct(**res), traces

def ConjugateGradient(init, iterround = 500):
    traces = [init.copy()]
    res = minimize(f, init, method='CG', callback=lambda x: traces.append(x), options={'maxiter': iterround})
    return res, traces

def Powell(init, iterround = 500):
    traces = [init.copy()]
    res = minimize(f, init, method='Powell', callback=lambda x: traces.append(x), options={'maxiter': iterround})
    return res, traces

def wh_2_xy(w, h):
    global scale
    return (scale*(w-origin[0]), scale*(origin[1]-h))

def xy_2_wh(x, y):
    global scale
    return (x/scale+origin[0], origin[1]-y/scale)


def colorize(diff_minimum):
    global maximum,minimum
    white = 0xfff
    ratio = (diff_minimum-minimum)/(maximum-minimum)
    large_2_small_scale_border = 0xff

    if ratio > 0.0:
        ratio = round(ratio*2, 1)/2
        R = int((1-ratio)*large_2_small_scale_border)
        G = int((1-ratio)*large_2_small_scale_border)
    else:
        ratio *= 10
        ratio = round(ratio*2, 1)/2
        R = int((1-ratio) * (0xff-large_2_small_scale_border)) + large_2_small_scale_border
        G = int((1-ratio) * (0xff-large_2_small_scale_border)) + large_2_small_scale_border

    color = R*256*256+G*256+0xcc
    color = "#%06x"%(color)
    return color


def draw_descend_region():
    global objective, width, height, radius, maximum, minimum

    minimum = 99999999999999999999999
    maximum = -99999999999999999999999
    print(obj, objective)
    #Recalculate the max and min
    for w in range(width):
        for h in range(height):
            mx, my = wh_2_xy(w, h)
            tmp = objective(mx, my)
            if maximum < tmp:
                maximum = tmp
            if minimum > tmp:
                minimum = tmp
    print(maximum, minimum)

    for w in range(0, width, radius):
        for h in range(0, height, radius):
            x,y = wh_2_xy(w, h)
            diff = objective(x, y)
            panel.create_oval(w-radius, h-radius, w+radius, h+radius, fill = colorize(diff), outline='')


def change_func_callback():
    global objective, obj, scale, CHANGE_FUNC
    obj_func = func_retrive.get()
    precision = precision_retrive.get()

    if "x1" not in obj_func or "x2" not in obj_func:
        messagebox.showinfo("Error", "No x1 or x2")
    else:
        #obj_func = obj_func.replace("e", "%.8f"%math.e)
        print(obj_func)
        x1 = Symbol('x1')
        x2 = Symbol('x2')
        obj = eval(obj_func)
        objective = lambdify((x1, x2), obj, 'numpy')
        scale = float(precision)
        CHANGE_FUNC = True
        draw_descend_region()

def analysis_change_callback(selection):
    global error_fig, value_fig,analysis_canvas, TOTAL_TRACES, terminate_flag, DRAW_FLAG, analysis_method_id
    analysis_method_id = TXT_2_ID[selection]

    error_fig.clf()
    DRAW_FLAG = False
    print(terminate_flag[analysis_method_id])
    if analysis_method_id in terminate_flag and terminate_flag[analysis_method_id]:
        iter_rnd = np.array([i+1 for i in range(len(TOTAL_TRACES[analysis_method_id]))])
        errors = []
        opt_value = f(TOTAL_TRACES[analysis_method_id][-1])

        for x in TOTAL_TRACES[analysis_method_id]:
            errors.append(np.abs(opt_value - f(x)))
        errors = np.array(errors)
        figplt = error_fig.add_subplot(111)
        figplt.set_xlabel("Iteration rounds")
        figplt.title.set_text("Errors on f(x) with optimum: %.3f"%opt_value)
        figplt.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

        for item in ([figplt.title, figplt.xaxis.label, figplt.yaxis.label] +
                     figplt.get_xticklabels() + figplt.get_yticklabels()):
            item.set_fontsize(fontsize)

        figplt.plot(iter_rnd, errors, color=COLOR[analysis_method_id])
        DRAW_FLAG = True
    print("change draw")
    analysis_canvas.draw()
    analysis_frame.update()

def draw_line_callback(event):
    global lines_records, objective, obj, scale, methods_int, methods, error_fig, analysis_var, \
        analysis_canvas, TOTAL_TRACES, terminate_flag, DRAW_FLAG, CHANGE_FUNC, analysis_method_id
    print("coordinates：",event.x,event.y, wh_2_xy(event.x,event.y))
    for lid in lines_records:
        panel.delete(lid)
    lines_records = []

    analysis_method_id = TXT_2_ID[analysis_var.get()]
    error_fig.clf()
    analysis_canvas.draw()
    analysis_frame.update()

    init = np.array(wh_2_xy(event.x,event.y))

    terminate_flag = {}
    for method_id in ALL_METHODS:
        if methods_int[method_id].get() == 1:
            terminate_flag[method_id] = False # All methods here will be displayed

    current_point = {}
    for method_id in terminate_flag.keys():
        current_point[method_id] = init.copy()
    cal_round = 5 # TODO

    TOTAL_TRACES = {}
    DRAW_FLAG = False
    while True:

        unconverged_list = []
        for method_id, flag in terminate_flag.items():
            if not flag:
                unconverged_list.append(method_id)
            else:
                if method_id == analysis_method_id and not DRAW_FLAG:
                    iter_rnd = np.array([i + 1 for i in range(len(TOTAL_TRACES[analysis_method_id]))])
                    errors = []
                    opt_value = f(TOTAL_TRACES[analysis_method_id][-1])

                    for x in TOTAL_TRACES[analysis_method_id]:
                        errors.append(np.abs(opt_value - f(x)))
                    errors = np.array(errors)

                    figplt = error_fig.add_subplot(111)
                    figplt.set_xlabel("Iteration rounds")
                    figplt.title.set_text("Errors on f(x) with optimum: %.3f"%opt_value)
                    figplt.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

                    for item in ([figplt.title, figplt.xaxis.label, figplt.yaxis.label] +
                                 figplt.get_xticklabels() + figplt.get_yticklabels()):
                        item.set_fontsize(fontsize)

                    figplt.plot(iter_rnd, errors, color=COLOR[analysis_method_id])

                    DRAW_FLAG = True
                    print("normal draw")
                    analysis_canvas.draw()
                    analysis_frame.update()

        TRACES = {}

        if len(unconverged_list) == 0:
            break

        loop_cnt = 0

        for method_id in unconverged_list:
            res, TRACES[method_id] = methods[method_id](current_point[method_id], iterround = cal_round)
            terminate_flag[method_id] = res.success
            if method_id not in TOTAL_TRACES:
                TOTAL_TRACES[method_id] = []
            TOTAL_TRACES[method_id].extend(TRACES[method_id][1:])

            current_point[method_id] = TRACES[method_id][-1].copy()

        while(True):
            paint_flag = False
            for method_id, traces in TRACES.items():
                if CHANGE_FUNC:
                    CHANGE_FUNC = False
                    break
                if loop_cnt < len(traces)-1:
                    p1 = xy_2_wh(traces[loop_cnt][0], traces[loop_cnt][1])
                    p2 = xy_2_wh(traces[loop_cnt+1][0], traces[loop_cnt+1][1])
                    lines_records.append(panel.create_line(p1[0],p1[1],p2[0],p2[1], fill=COLOR[method_id], width=3))
                    paint_flag = True

            if not paint_flag:
                break

            panel.pack()
            cavans_frame.update()
            loop_cnt += 1
            time.sleep(0.05)


methods = {BFGS: BFGS_method, ELS: exact_line_search, NEWTON: Newton, POWELL:Powell, NELDERMEAD:NelderMead, CG:ConjugateGradient}

if __name__ == "__main__":
    # Considering it is time-consuming to redraw the figures, make the window size fixed.
    # It can be adjusted before each run

    width = 800
    height = 600
    fontsize = width/100
    radius = 5
    scale = 0.01  # width pixels means the coordinates maximum is width*scale
    origin = (width / 2, height / 2)
    minimum = 99999999999999999999999
    maximum = -99999999999999999999999

    root = Tk()
    left_frame = Frame(master=root)
    right_frame = Frame(master=root)
    text_frame = Frame(master=left_frame)
    methods_frame = Frame(master=left_frame)

    analysis_frame = Frame(master=right_frame)
    cavans_frame = Frame(master=left_frame)
    minimum_point = (0, 0)

    alert_label = Label(text_frame, text="Enter the function(only contains x1 and x2):")
    func_retrive = Entry(text_frame, width=30)
    func_retrive.insert(0, sample2)
    func_enter = Button(text_frame, text="Change function",  fg="red", command=change_func_callback)

    setting_accuracy = Label(text_frame, text="Precision:")
    precision_retrive = Entry(text_frame, width=5)
    precision_retrive.insert(0, "0.01")


    methods_int = {}
    for i,method_id in enumerate(ALL_METHODS):
        methods_int[method_id] = IntVar()
        tmp_bt = Checkbutton(methods_frame, text=TXT[method_id], bg=COLOR[method_id], variable=methods_int[method_id])
        tmp_bt.pack(side=LEFT)
        tmp_bt.select()


    panel = Canvas(cavans_frame, width=width, height=height)

    analysis_label = Label(analysis_frame, text="Change the method analyzed below:", fg="red").pack(side=TOP, fill=BOTH)
    choices = [method_name for method_name in TXT.values()]
    analysis_var = StringVar()
    analysis_var.set('Newton-CG')

    analysis_choice = OptionMenu(analysis_frame, analysis_var, *choices, command=analysis_change_callback)
    analysis_choice.pack(side=TOP, fill=BOTH);

    error_fig = plt.Figure(figsize=(5, 4), dpi=100)

    figplt = error_fig.add_subplot(111)
    figplt.set_xlabel("Iteration rounds")
    figplt.title.set_text("Errors on f(x)")
    figplt.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

    for item in ([figplt.title, figplt.xaxis.label, figplt.yaxis.label] +
                 figplt.get_xticklabels() + figplt.get_yticklabels()):
        item.set_fontsize(fontsize)

    figplt.plot()

    analysis_canvas = FigureCanvasTkAgg(error_fig, master=analysis_frame)  # A tk.DrawingArea.
    analysis_canvas.draw()

    analysis_canvas.get_tk_widget().pack(side=TOP, ipadx=width/10)

    alert_label.grid(row=0, column=0)
    func_retrive.grid(row=0, column=1)
    setting_accuracy.grid(row=0, column=2)
    precision_retrive.grid(row=0, column=3)
    func_enter.grid(row=0, column=4)
    panel.pack(side=BOTTOM)

    draw_descend_region()

    lines_records = []

    panel.bind("<Button-1>",draw_line_callback)

    left_frame.pack(side=LEFT)
    right_frame.pack(side=RIGHT)
    text_frame.pack()
    text_frame.update()
    methods_frame.pack()
    analysis_frame.pack(side=RIGHT)
    cavans_frame.pack()
    mainloop()
