import numpy as np
import pandas as pd


def make_excel_table(res, file):
    res = np.transpose(np.asarray(res))
    df = pd.DataFrame(res)
    df.to_excel(file, sheet_name="Испытания")
