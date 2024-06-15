from bs4 import BeautifulSoup
import requests
import random

URL = "https://news.ycombinator.com/"
# then add number
PAGINATION = "https://news.ycombinator.com/?p="
# Randomize user_agent on startup or every 30 minutes
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
USER_AGENT_LIST = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.3", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.3", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.", "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.3"]

def run():
    random_index = random.randint(0, len(USER_AGENT_LIST))
    user_agent = USER_AGENT_LIST[random_index]
    headers = {'User-Agent': user_agent}
    try:
        page = requests.get(URL, headers=headers)
        soup = BeautifulSoup(page.content, 'html.parser')
        # print(soup.prettify())

        # td class="title" for title and <a href> inside of span class="titleline" for link
        # span class="age" for age of post
        title = soup.find("span", class_="titleline") # Should find_all instead
        link = title.find_next("a") # Should find_all_next instead
        age = soup.find("span", class_="age") # Should find_all instead
        print(title, age, link)

        # Extract text inside link
        # extract age from span maybe just text is better than date
        # as then i can run script every x hours, and if age is more than that
        # i can skip it, maybe i don't need the age if i just skip article
        # if it has same title and link
    except Exception as e:
        print("Failed to get page data.")
        """
        <td class="title">
         <span class="titleline">
          <a href="https://blogs.windows.com/windowsexperience/2024/06/07/update-on-the-recall-preview-feature-for-copilot-pcs/">
           Microsoft's Recall AI feature is now indefinitely delayed
          </a>
          <span class="sitebit comhead">
           (
           <a href="from?site=windows.com">
            <span class="sitestr">
             windows.com
            </span>
           </a>
           )
          </span>
         </span>
        </td>
       </tr>
       <tr>
        <td colspan="2">
        </td>
        <td class="subtext">
         <span class="subline">
          <span class="score" id="score_40683210">
           251 points
          </span>
          by
          <a class="hnuser" href="user?id=retskrad">
           retskrad
          </a>
          <span class="age" title="2024-06-14T18:03:09">
           <a href="item?id=40683210">
            1 hour ago
           </a>
          </span>
          <span id="unv_40683210">
          </span>
          |
          <a href="hide?id=40683210&amp;goto=news">
           hide
          </a>
          |
          <a href="item?id=40683210">
           130 comments
          </a>
         </span>
        </td>
       </tr>
       <tr class="spacer" style="height:5px">
       </tr>
       <tr class="athing" id="40682401">
        <td align="right" class="title" valign="top">
         <span class="rank">
          2.
         </span>
        </td>
        <td class="votelinks" valign="top">
         <center>
          <a href="vote?id=40682401&amp;how=up&amp;goto=news" id="up_40682401">
           <div class="votearrow" title="upvote">
           </div>
          </a>
         </center>
        </td>
        """
        return None

if __name__ == "__main__":
    run()
