from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

# -------------------------
# CONFIG
# -------------------------
JOB_TITLES = ["Software Engineer", "Data Scientist"]
LOCATIONS = ["Mumbai" , "Bangalore", "Delhi"]
MAX_JOBS_PER_SITE = 50

# -------------------------
# SETUP SELENIUM
# -------------------------
options = Options()
#options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def wait_and_scroll(driver, scroll_times=10, delay=4):
    """Scroll to load dynamic content"""
    for _ in range(scroll_times):
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
        time.sleep(delay)

# -------------------------
# INDEED
# -------------------------
def scrape_indeed(job, loc):
    rows = []
    url = f"https://in.indeed.com/jobs?q={job.replace(' ', '+')}&l={loc.replace(' ', '+')}"
    driver.get(url)
    time.sleep(8)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    cards = soup.select("div.cardOutline")[:MAX_JOBS_PER_SITE]

    for c in cards:
        title = c.select_one("h2.jobTitle span")
        # updated selector for company
        company = c.select_one("span[data-testid='company-name']")
        location = c.select_one("div.companyLocation")
        desc = c.select_one("div.job-snippet")
        link_tag = c.select_one("a.jcs-JobTitle")

        rows.append({
            "site": "Indeed",
            "title": title.text.strip() if title else "",
            "company": company.text.strip() if company else "Not specified",
            "location": location.text.strip() if location else loc,
            "description": desc.text.strip().replace("\n", " ") if desc else "",
            "link": "https://in.indeed.com" + link_tag["href"] if link_tag else ""
        })
    print(f"✅ Indeed: {len(rows)} jobs for {job} in {loc}")
    return rows


# -------------------------
# LINKEDIN
# -------------------------
def scrape_linkedin(job, loc):
    rows = []
    url = f"https://www.linkedin.com/jobs/search?keywords={job.replace(' ', '%20')}&location={loc.replace(' ', '%20')}"
    driver.get(url)
    wait_and_scroll(driver, scroll_times=4)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    cards = soup.select("div.base-card")[:MAX_JOBS_PER_SITE]

    for c in cards:
        title = c.select_one("h3.base-search-card__title")
        company = c.select_one("h4.base-search-card__subtitle")
        location = c.select_one("span.job-search-card__location")
        desc = c.select_one("p.job-search-card__snippet")
        link = c.select_one("a[href]")
        rows.append({
            "site": "LinkedIn",
            "title": title.text.strip() if title else "",
            "company": company.text.strip() if company else "",
            "location": location.text.strip() if location else loc,
            "description": desc.text.strip() if desc else "",
            "link": link["href"] if link else ""
        })
    print(f"✅ LinkedIn: {len(rows)} jobs for {job} in {loc}")
    return rows
    

# -------------------------
# TIMESJOBS
# -------------------------
def scrape_timesjobs(job, loc):
    rows = []
    url = f"https://www.timesjobs.com/candidate/job-search.html?searchType=Home_Search&from=submit&txtKeywords={job.replace(' ', '+')}&txtLocation={loc.replace(' ', '+')}"
    driver.get(url)
    time.sleep(6)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    cards = soup.select("li.clearfix.job-bx")[:MAX_JOBS_PER_SITE]

    for c in cards:
        title = c.select_one("h2 a")
        # updated company selector
        company = c.select_one("h3.joblist-comp-name")
        location = c.select_one("ul.top-jd-dtl li span")
        desc = c.select_one("ul.list-job-dtl.clearfix li span")
        link = title["href"] if title else ""

        rows.append({
            "site": "TimesJobs",
            "title": title.text.strip() if title else "",
            "company": company.text.strip() if company else "Not specified",
            "location": location.text.strip() if location else loc,
            "description": desc.text.strip() if desc else "",
            "link": link
        })
    print(f"✅ TimesJobs: {len(rows)} jobs for {job} in {loc}")
    return rows



# -------------------------
# RUN ALL SCRAPERS
# -------------------------
all_jobs = []
for job in JOB_TITLES:
    for loc in LOCATIONS:
        all_jobs += scrape_indeed(job, loc)
        all_jobs += scrape_linkedin(job, loc)
        all_jobs += scrape_timesjobs(job, loc)

driver.quit()

# -------------------------
# SAVE TO CSV
# -------------------------
df = pd.DataFrame(all_jobs)
df.drop_duplicates(subset=["link"], inplace=True)
df.to_csv("all_jobs.csv", index=False, encoding='utf-8-sig')
print(f"✅ Saved {len(df)} job listings to all_jobs.csv")
