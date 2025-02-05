import requests  # to send http requests
from bs4 import BeautifulSoup  # to parse html
import pandas as pd
import numpy as np
import cloudscraper
import time  # For adding delay between requests to avoid rate limiting

pages = [
    "https://sofifa.com/players?showCol%5B0%5D=pi&showCol%5B1%5D=ae&showCol%5B2%5D=hi&showCol%5B3%5D=wi&showCol%5B4%5D=pf&showCol%5B5%5D=oa&showCol%5B6%5D=pt&showCol%5B7%5D=bo&showCol%5B8%5D=bp&showCol%5B9%5D=gu&showCol%5B10%5D=vl&showCol%5B11%5D=wg&showCol%5B12%5D=rc&showCol%5B13%5D=ta&showCol%5B14%5D=cr&showCol%5B15%5D=fi&showCol%5B16%5D=he&showCol%5B17%5D=sh&showCol%5B18%5D=vo&showCol%5B19%5D=ts&showCol%5B20%5D=dr&showCol%5B21%5D=cu&showCol%5B22%5D=fr&showCol%5B23%5D=lo&showCol%5B24%5D=bl&showCol%5B25%5D=to&showCol%5B26%5D=ac&showCol%5B27%5D=sp&showCol%5B28%5D=ag&showCol%5B29%5D=re&showCol%5B30%5D=ba&showCol%5B31%5D=tp&showCol%5B32%5D=so&showCol%5B33%5D=ju&showCol%5B34%5D=st&showCol%5B35%5D=sr&showCol%5B36%5D=ln&showCol%5B37%5D=te&showCol%5B38%5D=ar&showCol%5B39%5D=in&showCol%5B40%5D=po&showCol%5B41%5D=vi&showCol%5B42%5D=pe&showCol%5B43%5D=cm&showCol%5B44%5D=td&showCol%5B45%5D=ma&showCol%5B46%5D=sa&showCol%5B47%5D=sl&showCol%5B48%5D=tg&showCol%5B49%5D=gd&showCol%5B50%5D=gh&showCol%5B51%5D=gc&showCol%5B52%5D=gp&showCol%5B53%5D=gr&showCol%5B54%5D=tt&showCol%5B55%5D=bs&showCol%5B56%5D=ir&showCol%5B57%5D=pac&showCol%5B58%5D=sho&showCol%5B59%5D=pas&showCol%5B60%5D=dri&showCol%5B61%5D=def&showCol%5B62%5D=phy%3D&offset=0"
]
for page in range(0, 5000, 60):
    pages.append(
        "https://sofifa.com/players?showCol%5B0%5D=pi&showCol%5B1%5D=ae&showCol%5B2%5D=hi&showCol%5B3%5D=wi&showCol%5B4%5D=pf&showCol%5B5%5D=oa&showCol%5B6%5D=pt&showCol%5B7%5D=bo&showCol%5B8%5D=bp&showCol%5B9%5D=gu&showCol%5B10%5D=vl&showCol%5B11%5D=wg&showCol%5B12%5D=rc&showCol%5B13%5D=ta&showCol%5B14%5D=cr&showCol%5B15%5D=fi&showCol%5B16%5D=he&showCol%5B17%5D=sh&showCol%5B18%5D=vo&showCol%5B19%5D=ts&showCol%5B20%5D=dr&showCol%5B21%5D=cu&showCol%5B22%5D=fr&showCol%5B23%5D=lo&showCol%5B24%5D=bl&showCol%5B25%5D=to&showCol%5B26%5D=ac&showCol%5B27%5D=sp&showCol%5B28%5D=ag&showCol%5B29%5D=re&showCol%5B30%5D=ba&showCol%5B31%5D=tp&showCol%5B32%5D=so&showCol%5B33%5D=ju&showCol%5B34%5D=st&showCol%5B35%5D=sr&showCol%5B36%5D=ln&showCol%5B37%5D=te&showCol%5B38%5D=ar&showCol%5B39%5D=in&showCol%5B40%5D=po&showCol%5B41%5D=vi&showCol%5B42%5D=pe&showCol%5B43%5D=cm&showCol%5B44%5D=td&showCol%5B45%5D=ma&showCol%5B46%5D=sa&showCol%5B47%5D=sl&showCol%5B48%5D=tg&showCol%5B49%5D=gd&showCol%5B50%5D=gh&showCol%5B51%5D=gc&showCol%5B52%5D=gp&showCol%5B53%5D=gr&showCol%5B54%5D=tt&showCol%5B55%5D=bs&showCol%5B56%5D=ir&showCol%5B57%5D=pac&showCol%5B58%5D=sho&showCol%5B59%5D=pas&showCol%5B60%5D=dri&showCol%5B61%5D=def&showCol%5B62%5D=phy%3D&offset="
        + str(page)
    )

pages


def getAndParseURL(url):
    # use cloudscraper to bypass cloudflare protection
    scraper = cloudscraper.create_scraper()
    result = scraper.get(url, headers={"User-Agent": "Mozilla/5.0"})

    # return the html from the url which our sofifa pages
    soup = BeautifulSoup(result.text, "html.parser")
    return soup


players = []
# loop through each page url which we already have
for page in pages:
    html = getAndParseURL(page)
    table = html.find("table")  # Try to find the table containing player data

    if (
        table is not None
    ):  # table is the ovverall container of player data, holds all rows and columns
        # Loop through each row (excluding the header row)
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) > 0:  # Only process rows that have columns
                player = {"name": cols[1].get_text().strip()}  # Get player name
                headers = [
                    th.get_text().strip() for th in table.find("tr").find_all("th")
                ]  # Extract the headers from the table (defines the strucuter of columns like Name, age , overall)
                for i, col in enumerate(cols[2:], start=2):
                    if i < len(headers):  # ensure index is valid
                        player[headers[i]] = col.get_text().strip()

                players.append(player)
    else:
        print(
            f"Table not found on page: {page}"
        )  # this is just for time saving, i waited too long for the error before.

    time.sleep(1)  # avoid getting rate-limited

players

df = pd.DataFrame(players)
df

# with pd.option_context('display.max_columns', None, 'display.max_rows', 5):
# display(df.head(5))

df.describe()
