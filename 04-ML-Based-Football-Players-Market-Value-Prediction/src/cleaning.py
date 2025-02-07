import pandas as pd 
import numpy as np 
df = pd.read_csv("../data/raw/scraped_players_data.csv")
df.head()
# Check for missing values
df.isnull().sum()
pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None) 
pd.set_option('display.expand_frame_repr', False)
df.drop(columns = ['Unnamed: 64'], inplace=True)
df.columns = df.columns.str.strip()
df.dtypes
# check duplicates
df.duplicated().sum()
df.drop_duplicates(inplace=True)


df[['Name', 'Position']] = df['name'].str.split(r'\s+(?=[A-Z]{2,3}\b)', n=1, expand=True) 
df = df.drop('name', axis=1) 

# r'\s+(?=[A-Z]{2,3}\b means split the string on the first space before a capital letter that is followed by 2 or 3 capital letters ex, L.Messi CAM RW ---> L.Messi, CAM RW
df.head(5)
df = df.drop('Position', axis=1)  # drop position column as we already have best position column
df.head(4)
# Reorder columns to place 'Name' first
cols = ['Name'] + [col for col in df.columns if col != 'Name']
df = df[cols]
df.head()


df[['Team', 'Contract Duration']] = df['Team & Contract'].str.split('\n', n=1, expand=True)
df = df.drop('Team & Contract', axis=1)  # Drop the original column
df.head(3)


text_cols = df.select_dtypes(include='object').columns # select df with objects like text/string columns
for col in text_cols:
    df[col] = df[col].str.replace('\n', ' ', regex=False) # /n treted as literla, not regex pattern
    df[col] = df[col].str.strip() # remove leading and trailing whitespaces|
    df.head(6)
df.dtypes



text_cols = df.select_dtypes(include='object').columns
for col in text_cols:
    # Apply cleaning only if the column contains values with '+' signs (e.g., '75+1')
    if df[col].str.contains(r'\d+\+').any():  # check if there's any + sign in the column
        # Extract the base number before '+' and convert it to int
        df[col] = df[col].astype(str).str.extract(r'^(\d+)').astype(int)

df.head(5)


df['Height_cm'] = df['Height'].str.extract(r'(\d+)cm').astype(int)
df['Weight_kg'] = df['Weight'].str.extract(r'(\d+)kg').astype(int)
df = df.drop(['Height', 'Weight'], axis=1)
df.head(3)    
    

def convert_currency(value):
    value = str(value).replace('€', '') # remove euro symbol
    if 'M' in value:
        return float(value.replace('M', '')) * 1e6 # eg: convert 5.5M to 5.5 to 5500000
    elif 'K' in value:
        return float(value.replace('K', '')) * 1e3 # eg: convert 5.5K to 5.5 to 5500
    else:
        return float(value) # eg: return as it is by converting to float

df['Value'] = df['Value'].apply(convert_currency)
df['Wage'] = df['Wage'].apply(convert_currency)
df['Release clause'] = df['Release clause'].apply(convert_currency)

# define new column order
column_order = [
    'Name', 'ID', 'Age', 'Best position', 'Best overall', 'Overall rating', 'Potential', 'Growth', 'foot',
    'Team', 'Contract Duration', 'Value', 'Wage', 'Release clause',
    'Height_cm', 'Weight_kg', 'Acceleration', 'Sprint speed', 'Agility', 'Reactions', 'Balance', 'Stamina', 'Strength', 'Jumping',
    'Total attacking', 'Crossing', 'Finishing', 'Heading accuracy', 'Short passing', 'Volleys',
    'Total skill', 'Dribbling', 'Curve', 'FK Accuracy', 'Long passing', 'Ball control',
    'Total defending', 'Defensive awareness', 'Standing tackle', 'Sliding tackle', 'Interceptions', 'Aggression',
    'Total goalkeeping', 'GK Diving', 'GK Handling', 'GK Kicking', 'GK Positioning', 'GK Reflexes',
    'Total mentality', 'Att. Position', 'Vision', 'Penalties', 'Composure',
    'Total power', 'Shot power', 'Long shots',
    'Total stats', 'Base stats', 'International reputation',
    'Pace / Diving', 'Shooting / Handling', 'Passing / Kicking', 'Dribbling / Reflexes', 'Defending / Pace'
]

# reorder the dataframe
df = df[column_order]
df.head()


print(df.isnull().sum().sum())
(df == '').sum().sum()
df.replace(['', 'Unknown', 'N/A', '-', 0], np.nan, inplace=True)
df.head(5)
# Let's see the height outliers
df[(df['Height_cm'] < 160) | (df['Height_cm'] > 205)]
#Let's see the weight outliers
df[(df['Weight_kg'] < 50) | (df['Weight_kg'] > 100)]
# Let's see the age outliers
df[(df['Age'] < 17) ]
df[(df['Age'] > 40)]


df['On Loan'] = df['Contract Duration'].str.contains('On loan', case=False).astype(int)

df['Contract End Year'] = df['Contract Duration'].str.extract(r'(\d{4})\D*$')

df['Contract End Year'] = df['Contract End Year'].fillna(0).astype(int)

df = df.drop('Contract Duration', axis=1)
df.head(3)