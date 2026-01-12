import requests
import json
import datetime
import pandas as pd
from tabulate import tabulate

# Replace these with your actual values
token = "dummy"
headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {token}",
}

org = "finance-technology-cill"
url = f"https://gecgithub01.walmart.com/api/v3/orgs/{org}/repos"

week_wise_data = []

def week_wise_stats(owner: str, repo: str):
    url = f"https://gecgithub01.walmart.com/api/v3/repos/{owner}/{repo}/stats/commit_activity"
    response = requests.get(url, headers=headers)
    json_data = response.json()
    for jd in json_data:
        week_date = datetime.datetime.fromtimestamp(jd["week"]).strftime('%Y-%m-%d')
        week_wise_data.append({"repo": repo, "week_date": week_date, "total_commits": jd["total"]})

with open('repos.txt', 'r') as file:
    repos = [line.strip() for line in file.readlines()]

for repo in repos:
    print(repo)
    if repo == "cill-reporting-services":
        print(f"skipping {repo}")
    else:
        week_wise_stats(owner=org, repo=repo)

df = pd.DataFrame(week_wise_data)
df = df.sort_values(by=["repo", "week_date"], ascending=[True, True])
df = df.pivot(index="repo", columns="week_date", values="total_commits").fillna(0)
df.columns = [str(x) for x in df.columns]

df['all_commits'] = df.iloc[:, 1:].sum(axis=1)
df['monthly_avg'] = (df[['all_commits']]/12).round(2)
df['weekly_avg'] = (df[['all_commits']]/52).round(2)

print(tabulate(df, headers="keys", tablefmt="psql"))
df.to_excel('results123.xlsx', header=True)
