# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:04:11 2021

@author: sujitk
"""

import pyodbc
import pandas as pd
conn = pyodbc.connect("Driver={SQL Server};"
                      "Server=DataScience01;"
                      "Database=DS01;"
                      "Trusted_Connection=yes;")
#cursor = conn.cursor()
qry =  """select top (1000) *
        from x_rate_history
    """

mcleod = pd.read_sql("""select top (1000) *
from x_rate_history
    """,conn)
    
    
mcleod.start_market = mcleod.start_market.str.strip()
mcleod.dest_market = mcleod.dest_market.str.strip()

df = mcleod[~((mcleod.start_market == mcleod.dest_market)&
	(mcleod.pay_distance > 500))].sort_values(["start_market","equipment_id", "actual_arrival"])
df = df.reset_index(drop=True)

df["rpm"] = df["freight_chg"]/df["pay_distance"]  
df_start1 = df.groupby(["start_market","dest_market","equipment_id"]).agg({"rpm":"mean","pay_distance":"mean",
			"spot_freight_chg":"mean","freight_chg":"mean"})
df_start2 = df.groupby(["start_market","dest_market","equipment_id"]).size()

df_start = pd.concat([df_start1,df_start2],axis = 1).reset_index()
df_start.columns = ["start_market","dest_market","equipment_id","rpm","pay_distance","spot_freight_chg","freight_chg","count_"]

OutputDataSet  = df_start







