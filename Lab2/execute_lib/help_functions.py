def center(x1, x2, in1, in2):
    x = (x2 + x1) / 2
    i = (in2 + in1) / 2
    change = i - x
    return x1 + change, x2 + change