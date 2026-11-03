"""Audit a supplied NYC 311 CSV without changing it. Usage: python 06_audit_sample.py path.csv"""
import sys
import pandas as pd
p=sys.argv[1] if len(sys.argv)>1 else 'data/sample_311.csv'
df=pd.read_csv(p,low_memory=False)
df=df.rename(columns={'problem':'complaint_type','problem_detail':'descriptor'})
print('Rows:',f'{len(df):,}')
print('Columns:',len(df.columns),list(df.columns))
for c in ['created_date','closed_date','due_date']:
 if c in df:
  x=pd.to_datetime(df[c],errors='coerce')
  print(f'{c}: valid={x.notna().sum():,}; min={x.min()}; max={x.max()}')
if 'unique_key' in df: print('Unique keys:',df.unique_key.nunique(),'duplicates:',df.unique_key.duplicated().sum())
for c in ['agency','complaint_type','borough','status']:
 if c in df: print(f'{c}: {df[c].nunique(dropna=True)} distinct; missing={df[c].isna().sum():,}')
if {'created_date','closed_date'} <= set(df.columns):
 cr=pd.to_datetime(df.created_date,errors='coerce'); cl=pd.to_datetime(df.closed_date,errors='coerce')
 print('Closed before created:',int((cl.notna()&cr.notna()&(cl<cr)).sum()))
