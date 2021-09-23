# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 13:26:21 2020

@author: sujitk
"""

import pyodbc
import pandas as pd
conn = pyodbc.connect("Driver={SQL Server};"
                      "Server=DataScience01;"
                      "Database=DS01;"
                      "Trusted_Connection=yes;")
cursor = conn.cursor()


df = pd.read_sql(
 """SELECT 
         dest_market_id
        ,origin_market_id
        ,hazmat
        ,freight_chg
        ,pay_distance
        ,origin_yrweek
        ,spot_freight_pm
        ,fsc
        ,spot_fsc
        ,spot_other_chg
        ,other_chg
        ,spot_tot_chgs,
        tot_chgs
        ,spot_freight_chg
        ,freight_pm
        ,exchange_type
        ,origin_yrweek
        ,exclude_from_spot
        ,exclude_from_rate
        ,equipment_type_id
        ,dest_state
        ,origin_state
    FROM [DS01].[dbo].[x_rate_history] 
    where exclude_from_rate = 'N'
    
    union 
    
    SELECT 
         dest_market_id
        ,origin_market_id
        ,hazmat
        ,freight_chg
        ,pay_distance
        ,origin_yrweek
        ,spot_freight_pm
        ,fsc
        ,spot_fsc
        ,spot_other_chg
        ,other_chg
        ,spot_tot_chgs,
        tot_chgs
        ,spot_freight_chg
        ,freight_pm
        ,exchange_type
        ,origin_yrweek
        ,exclude_from_spot
        ,exclude_from_rate
        ,equipment_type_id
        ,dest_state
        ,origin_state
    FROM [DS01].[dbo].[vw_x_rate_history_YTD] 
    where exclude_from_rate = 'N'
    """,conn)
