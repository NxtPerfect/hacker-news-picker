package gosrc

import "testing"

func TestRequestsToHackerNews(t *testing.T) {
	resp, err := requestPages()
	if err != nil {
		t.Fatalf(`Requesting pages failed %q`, err)
	}
}
