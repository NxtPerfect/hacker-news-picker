from bs4 import BeautifulSoup
import requests

URL = "https://news.ycombinator.com/"
PAGINATION = "https://news.ycombinator.com/?p=" # then add number
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"

def run():
    headers = {'User-Agent': USER_AGENT}
    try:
        page = requests.get(URL, headers=headers)
        soup = BeautifulSoup(page.content, 'html.parser')
        # print(soup.prettify())

        # td class="title" for title and <a href> inside of span class="titleline" for link
        # span class="age" for age of post
        title = soup.find("span", class_="titleline")
        link = title.find_next("a")
        age = soup.find("span", class_="age")
        print(title, age, link)
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
