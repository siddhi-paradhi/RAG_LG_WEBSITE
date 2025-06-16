import scrapy

class CiscoSpider(scrapy.Spider):
    name = "cisco"
    allowed_domains = ["www.cisco.com"]
    start_urls = [
        "https://www.cisco.com/c/en/us/support/index.html",
    ]

    def parse(self, response):
        for link in response.css("a::attr(href)").getall():
            if "/c/en/us/support/" in link:
                full_url = response.urljoin(link)
                yield scrapy.Request(full_url, callback=self.parse_article)

    def parse_article(self, response):
        content = " ".join(response.css("main *::text").getall()).strip()
        if content:
            yield {
                "url": response.url,
                "content": content
            }
