package gosrc

import (
	"net/http"

	"golang.org/x/net/html"
)

const (
	CSV_PATH = "../data/news_go.csv"
	URL      = "https://news.ycombinator.com/?p="
)

var (
	USER_AGENT_LIST = [...]string{
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.3",
		"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.3",
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.",
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.",
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.3"}

	PROXIES_LIST = [...]string{
		"http://155.94.241.130:3128",
		"http://128.199.202.122:3128",
		"http://198.44.255.3:80",
		"http://31.220.78.244:80",
		"http://50.174.145.8:80",
		"http://58.234.116.197:80",
		"http://65.109.189.49:80",
		"http://123.30.154.171:7777",
		"http://50.168.163.177:80",
		"http://62.99.138.162:80",
		"http://91.92.155.207:3128",
		"http://85.8.68.2:80",
		"http://47.74.152.29:8888",
		"http://83.1.176.118:80",
		"http://167.102.133.111:80",
		"http://50.207.199.84:80",
		"http://103.163.51.254:80",
		"http://50.172.75.126:80",
		"http://211.128.96.206:80",
		"http://51.254.78.223:80"}
)

func run() {
	page, err := requestPages()
	if err != nil {
		panic("Failed to get pages")
	}
	tokenizer := html.NewTokenizer(page.Body)
	parsedPages := parsePages(page, tokenizer)
	saveToCsv(parsedPages)
}

func requestPages() (*http.Response, error) {
	resp, err := http.Get(URL)
	defer resp.Body.Close()
	return resp, err
}

func parsePages(page *http.Response, tokenizer html.Tokenizer) {
	// articles = soup.find_all("span", class_="titleline")
	// link = article.find_next("a")
	// # Find title
	// title = link.contents[0]
	// # Get link url
	// link = link["href"]
	return
}

func saveToCsv(pages) {
	return
}
