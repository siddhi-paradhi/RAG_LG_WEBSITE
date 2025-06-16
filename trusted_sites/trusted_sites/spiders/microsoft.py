import scrapy

class MicrosoftSpider(scrapy.Spider):
    name = "microsoft"
    allowed_domains = ["learn.microsoft.com"]
    start_urls = [
        "https://learn.microsoft.com/en-us/azure/",
    ]

    def parse(self, response):
        for link in response.css("a::attr(href)").getall():
            if link.startswith("/en-us/azure/"):
                full_url = response.urljoin(link)
                yield scrapy.Request(full_url, callback=self.parse_article)

    def parse_article(self, response):
        content = " ".join(response.css("main *::text").getall()).strip()
        if content:
            yield {
                "url": response.url,
                "content": content
            }
