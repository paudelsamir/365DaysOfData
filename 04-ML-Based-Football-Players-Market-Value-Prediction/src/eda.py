# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Day 4: EDA

# ![eda_based_on_marketvalue.png](attachment:eda_based_on_marketvalue.png)

# Let's GET started before diving into finding patterns. 

# > **Note:** This EDA will primarily focus on market value analysis rather than individual player analysis.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/processed/cleaned_data.csv")

df.head()

pd.set_option('display.max_rows', None) 
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)  
pd.set_option('display.max_colwidth', None)


df.head(5)

df.describe()

# # Performing Analysis

# value counts for players
print(df['Best position'].value_counts())
print(df['foot'].value_counts())
print(df['Team'].value_counts())


# correlation matrix
numerical_df = df.select_dtypes(include=[np.number])
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


df.head(5)

df[df['Name'].str.contains('Lamine Yamal', case=False, na=False)]

# market value: fluctates based on player form, performance, transfer rumors (We have growth/ potential to determine) <br>
# wage: means salary for a player 

# pairplot
sns.pairplot(df[['Value', 'Age', 'Overall rating','Best position']])
plt.show()


plt.figure(figsize=(10,6))
sns.histplot(df['Overall rating'], bins=30, kde=True, color='red')
plt.title('Distribution of Players with ratings')
plt.show()

# distribution of market values
sns.histplot(df['Value'], kde=True)
plt.show()


# we can see the left skewed distribution here. we can apply log transofrmation for better visualizeiton<br>
# This makes values like €100M and €1M less extreme by applying a logarithmic scale, improving analysis or prediction accuracy.

# +
df['Log_Value'] = df['Value'].apply(lambda x: np.log(x + 1))

# Plot the transformed values
sns.histplot(df['Log_Value'], kde=True)
plt.title('Log Transformed Market Values')
plt.show()
# -

print(f"Original Value Skewness: {df['Value'].skew()}")
print(f"Log-Transformed Value Skewness: {df['Log_Value'].skew()}")


# +
sns.scatterplot(x=df['Age'], y=df['Log_Value'])
plt.title('Age vs Log-Transformed Market Value')
plt.show()

sns.scatterplot(x=df['Potential'], y=df['Log_Value'])
plt.title('Potential vs Log-Transformed Market Value')
plt.show()

# -

# bar plot for Best position distribution
sns.countplot(x='Best position', data=df)
plt.xticks(rotation=90)
plt.show()


# >As my goal is to predict the market value of players, I will be focusing on the following observations and then documenting each finding during my analysis:

# 1. Market Value by positions:

sns.boxplot(x='Best position', y='Value', data=df)
plt.xticks(rotation=90)

# conclusion: Positions like attackers (e.g., ST) might have higher market values, while defenders (e.g., CB) might have lower values.

# 2. Market Value Vs Potential

sns.scatterplot(x='Potential', y='Value', data=df)

# +
import plotly.express as px

# Assuming 'Log_Value' is the log-transformed market value
fig = px.scatter_3d(df, x='Age', y='Potential', z='Value', 
                    color='Log_Value',  # Color by market value (log-transformed)
                    labels={'Age': 'Age', 'Potential': 'Potential', 'Log_Value': 'Log Market Value'},
                    title="3D Scatter Plot of Age, Potential, and Log Market Value")

fig.show()
# -

# 3. Market Value and positions

# group by position and calculate mean value
df.groupby('Best position')['Value'].mean().sort_values(ascending=False)
.3

# wingers market values seems higher 

import plotly.express as px
fig = px.scatter(df[df['Best position'].isin(['LW', 'RW'])], x='Pace / Diving', y='Value', 
                 title='Pace vs Market Value for Wingers', labels={'Pace / Diving': 'Pace', 'Value': 'Market Value'},
                 trendline='ols')  
fig.show()


# > i think i have to create a list of questions with the help of copilot and try to extract the conclusion from them. This would be the perfect Visualizations as well as the perfect conlcusion

# # **overall dataset insights**

# ### what are the **top 10 most valuable players**?

df.head(5)

df.sort_values(by='Value', ascending=False).head(10)

# ### how does **market value vary by position** (e.g., are strikers more expensive than defenders)?

# +
fig = px.box(df, x='Best position', y='Value', 
             title="Market Value Distribution by Position",
             labels={'Best position': 'Player Position', 'Value': 'Market Value (€)'},
             color='Best position',
             points= 'all')  # Do not show individual points (outliers)

fig.show()


# +
avg_value_by_position = df.groupby('Best position')['Value'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(x='Best position', y='Value', data=avg_value_by_position, palette='viridis')
plt.title("Average Market Value by Position")
plt.xlabel("Player Position")
plt.ylabel("Average Market Value (€)")
plt.xticks(rotation=90)
plt.show()
# -

# ### which teams have the **highest average market value**?

avg_value_by_team =  df.groupby('Team')['Value'].mean().sort_values(ascending=False)

fig = px.bar(avg_value_by_team, x=avg_value_by_team.index, y=avg_value_by_team.values, title='Average Market Value by Team', labels={'x': 'Team', 'y': 'Average Market Value (€)'}, color=avg_value_by_team.values, color_continuous_scale='viridis')
fig.show()

# Team with highest average market value: Real Madrid with an average market value of 51M euro.

# ### what’s the **distribution of market values** (is it skewed towards a few expensive players)?

sns.histplot(df['Value'], kde=True)

# The distribution of market values is likely right-skewed, indicating that most players have lower market values, while a small group of expensive players

# ### how does **age correlate with market value** (are younger players generally worth more)?

df.head(5)

# +
import plotly.express as px

# Create a 3D scatter plot to visualize the correlation between age, market value, and potential
fig = px.scatter_3d(df, x='Age', y='Value', z='Potential',
                    title="Age vs Market Value vs Potential",
                    labels={'Age': 'Age (years)', 'Value': 'Market Value (€)', 'Potential': 'Potential Rating', 'Name': 'Name'}, 
                    color='Age')

fig.show()

# -

# Typically, younger players with higher potential tend to have higher market values, but as age increases, the market value might decrease unless they perform exceptionally well.

# # **player attributes vs. market value**

# ### how does a player’s **overall rating affect their market value**?

# +
fig = px.scatter(df, x='Overall rating', y='Value',
                 title="Overall Rating vs Market Value",
                 labels={'Overall rating': 'Overall Rating', 'Value': 'Market Value (€)'},
                 color='Age',
                 trendline='ols')  # Add a trendline    

fig.show()
# -

# players with higher overall ratings tend to have higher market values, though there may be exceptions based on age, potential, or market dynamics.

# ### which individual attributes (e.g., pace, stamina, strength) **correlate the most with market value**?

df.head(5)

# +
attributes = ['Sprint speed', 'Stamina', 'Strength', 'Acceleration', 'Sprint speed', 'Agility', 'Reactions', 'Balance', 'Jumping', 'Dribbling']
corr_matrix = df[attributes + ['Value']].corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title("Correlation between Player Attributes and Market Value")
plt.show()

# +
top_correlated_attributes = ['Sprint speed', 'Acceleration', 'Dribbling', 'Reactions', 'Strength']

for attr in top_correlated_attributes:
    fig = px.scatter(df, x=attr, y='Value', 
                     title=f"{attr} vs Market Value",
                     labels={attr: f'{attr}', 'Value': 'Market Value (€)'},
                     color=attr, color_continuous_scale='Viridis')
    fig.show()

# -

# Player with higher dribbling and higher reaction tend to have more market value

# ### does **international reputation (1-5 stars) impact market value**?

# +
fig = px.box(df, x='International reputation', y='Value',
             title="Market Value Distribution by International Reputation",
             labels={'International reputation': 'International Reputation (Stars)', 'Value': 'Market Value (€)'},
             color='International reputation', points='all')

fig.show()

# -

# players with higher reputation tend to have higher market values, and if there are any outliers.

# ### how do **potential ratings** compare to market value (are high-potential players priced higher)?

# +
fig = px.scatter(df, x='Potential', y='Value',
                 title="Potential Rating vs Market Value",
                 labels={'Potential': 'Potential Rating', 'Value': 'Market Value (€)'},
                 color='Potential')

fig.show()
# -

# Generally, players with higher potential ratings tend to have higher market values, especially if they're young and have room for development. However, this may not always hold true if their current performance does not align with their potential.

# ### do **physical attributes (height, weight, strength) play a role in market value**?

# +
import seaborn as sns
import matplotlib.pyplot as plt

physical_attributes = ['Height_cm', 'Weight_kg', 'Strength']
corr_matrix = df[physical_attributes + ['Value']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title("Correlation between Physical Attributes and Market Value")
plt.show()


# +
import plotly.express as px

physical_attributes = ['Height_cm', 'Weight_kg', 'Strength']

for attr in physical_attributes:
    fig = px.scatter(df, x=attr, y='Value', 
                     title=f"{attr} vs Market Value",
                     labels={attr: f'{attr}', 'Value': 'Market Value (€)'},
                     color=attr, color_continuous_scale='Viridis')
    fig.show()

# -

#

# # **position-specific insights**

#
# ### are **attacking midfielders (CAM/CM) more valuable than defensive midfielders (CDM/CM)?**

# +
import plotly.express as px

midfielders = df[df['Best position'].isin(['CAM', 'CM', 'CDM'])]

fig = px.violin(midfielders, x='Best position', y='Value', box=True, points="all",
                title="Market Value Distribution: CAM/CM vs CDM",
                labels={'Best position': 'Midfield Position', 'Value': 'Market Value (€)'},
                color='Best position')

fig.show()

# -

# CDMs generally have lower market values, but elite defensive midfielders may still command high prices.

# ### how does **pace affect wingers' (LW/RW) market value**?

# +
wingers = df[df['Best position'].isin(['LW', 'RW'])]

fig = px.scatter(wingers, x='Pace / Diving', y='Value', 
                 title="Pace vs Market Value for Wingers (LW/RW)",
                 labels={'Pace / Diving': 'Pace Rating', 'Value': 'Market Value (€)'},
                 color='Pace / Diving', color_continuous_scale='Viridis',
                 trendline="ols") 

fig.show()
# -

# ### do goalkeepers follow the same market trends as outfield players?

# +
fig = px.violin(df, x='Position Type', y='Value', box=True, points="all",
                title="Market Value Spread: Goalkeepers vs Outfield Players",
                labels={'Position Type': 'Player Type', 'Value': 'Market Value (€)'},
                color='Position Type')

fig.show()

# -

# overall, outfield players command higher market values, while only a few elite goalkeepers reach top-tier valuations.

# # **contract & transfer market impact**

# ### does a player's **contract end year affect their market value** (e.g., do players with 1 year left have lower values)?

# +
fig = px.box(df, x='Contract End Year', y='Value',
             title="Market Value by Contract End Year",
             labels={'Contract End Year': 'Contract End Year', 'Value': 'Market Value (€)'},
             color='Contract End Year')

fig.show()

# -

# Players with shorter contract lengths (1 year left) often have lower market values because clubs risk losing them for free.
# Longer contracts (3+ years) tend to keep market values high as clubs have more control over transfers.
# Top players can still maintain high values regardless of contract length due to demand and reputation.

# ### are **players on loan priced differently** compared to permanent squad members?

# +
import plotly.express as px

fig = px.violin(df, x='Value', y='On Loan', orientation='h', box=True, 
                title="Density of Market Values: Loan vs Permanent Players",
                labels={'On Loan': 'On Loan?', 'Value': 'Market Value (€)'},
                color='On Loan')

fig.show()

# -

# loaned players often have lower values, while permanent players have a wider spread, including high-value stars.

# # Now i am going to analyze the talents like Lamine and Endrik for just fun purpose

players = ['Lamine Yamal', 'Endrick']
df[df['Name'].isin(players)]

df_stars = df[df['Name'].isin(players)]

# +

import plotly.express as px

attributes = ['Sprint speed', 'Dribbling', 'Long passing', 'Ball control', 'Total defending']
df_stars_renamed = dfj_stars.rename(columns={'Value': 'Market Value'})
df_melted = df_stars_renamed.melt(id_vars=['Name'], value_vars=attributes, var_name='Attribute', value_name='Value')

# Radar plot
fig = px.line_polar(df_melted, r='Value', theta='Attribute', color='Name',
                     title="Skill Breakdown of Future Stars", line_close=True)

fig.show()


# +
import numpy as np

# Estimate future market value (simple linear projection)
df_stars['Future Value'] = df_stars['Value'] * (df_stars['Potential'] / df_stars['Overall rating'])

fig = px.bar(df_stars, x='Name', y=['Value', 'Future Value'], barmode='group',
             title="Current vs Projected Future Market Value",
             labels={'value': 'Market Value (€)', 'variable': 'Stage'},
             color_discrete_map={'Value': 'blue', 'Future Value': 'red'})

fig.show()

# -

# Conclusion: In today's EDA, we focused on players' market value rather than finding the best talents and their potential. We analyzed the distribution of market value, the correlation between different attributes and market value, average market value by position and team, and market value distribution by international reputation. Additionally, we examined the correlation between physical attributes and market value, market value distribution for different positions, and the market value distribution for loan and permanent players. We also explored the skill breakdown of future stars, the potential rating vs. age for future stars, and estimated the future market value of these stars.

# That's it for today, Wonderful 6 hours of my life !!!! GoodBye
