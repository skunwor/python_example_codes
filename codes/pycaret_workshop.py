# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:03:16 2020

@author: sujitk
"""


import pyodbc
import pandas as pd
conn = pyodbc.connect("Driver={SQL Server};"
                      "Server=DataScience01;"
                      "Database=DS01;"
                      "Trusted_Connection=yes;")
cursor = conn.cursor()
#cursor.execute(
sql = '''SELECT dest_market_id,origin_market_id,
            dest_city_name,origin_city_name,hazmat,freight_chg, pay_distance,
            freight_pm,exchange_type,origin_yrweek,dest_state,origin_state 
        FROM [DS01].[dbo].[vw_x_rate_history_YTD] 
        WHERE dest_state is not null and origin_state is not null and freight_chg is not null'''   
rate_1 = pd.read_sql(sql,conn)

unique_market_ids = rate_1
