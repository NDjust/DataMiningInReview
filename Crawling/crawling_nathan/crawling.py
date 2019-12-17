from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, InvalidSelectorException

import time


def load_chrome_browser():
    """
    Load chrome brawser function. (to remove overlap code)

    :return: Chrome browser Object.
    """
    chrome_options = Options()
    browser = webdriver.Chrome(
        './webdriver/chrome/chromedriver.exe', options=chrome_options)
    browser.set_window_size(1920, 1280) # 윈도우 사이즈를 맞춰서 크롤링하기 쉽게 만들기.

    return browser


def get_searching_url(keyword):
    """
    Get searching keyword site url by using Google chrome drives.
    (+ if you don't use chrome driver, you could not scrape page html source)

    :param keyword: input to search keyword.
    :return: searching page url.
    :rtype: string
    """
    browser = load_chrome_browser()
    browser.get("https://www.mangoplate.com/")

    # input keyword in search box.
    input_element = browser.find_element_by_name('main-search')
    input_element.send_keys(keyword)

    # await moment & Search keyword.
    browser.implicitly_wait(3)
    browser.find_element_by_xpath(
        "/html/body/main/article/header/div/fieldset/input").send_keys(Keys.ENTER)

    # get current url.
    searching_url = browser.current_url

    return searching_url


def save_link(keyword, file_name):
    """
    Save all restaurant links to the input keyword.

    :param keyword: input keyword.
    :param file_name: input save file name.
    :return: None
    """
    # get all page urls.
    page_urls = ["https://www.mangoplate.com/search/{}?keyword={}&page={}".format(
        keyword, keyword, i) for i in range(1, 11)]

    # Open searching all pages.
    for url in page_urls:
        browser = load_chrome_browser()
        browser.get(url)
        html = browser.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # get all restaurants links
        links = soup.select(
            "figure > figcaption > div > a"
        )
        # Save all restaurants link at each page.
        for link in links:
            target_link = "https://www.mangoplate.com/" + link.get('href')
            with open("./{}.txt".format(file_name), 'a') as f:
                f.write("{}\n".format(target_link))

    return 0


def click_view_button(browser):
    """
    Click view more button.

    :param browser: target browser.
    :return: None
    """
    # 더보기 버튼이 없을때 까지 반복.
    while browser.find_element_by_css_selector(
            'div.RestaurantReviewList__MoreReviewButton').text:
        try:
            sample = browser.find_element_by_css_selector(
                'div.RestaurantReviewList__MoreReviewButton')
            browser.execute_script("arguments[0].click();", sample)  # 자바 명령어 실행
        except:
            break
        time.sleep(1)  # 잠깐 Pause 해야 다음 더보기 버튼이 로드 되면서 끝까지 버튼이 제자리에서 계속해서 누르는 것을 방지.


def get_review(link):
    """
    Get review data.

    :param link: target link.
    :return: review data (review text, review score)
    :rtype: list
    """
    browser = load_chrome_browser()
    browser.get(link)

    click_view_button(browser)

    review_counts = int(browser.find_element_by_class_name(
        "RestaurantReviewList__AllCount").text)
    reviews = list()

    # scraping review data.
    for i in range(1, review_counts + 1):
        review_score = []

        # get review text.
        try:
            review_score.append(browser.find_element_by_css_selector(
                "section.RestaurantReviewList > ul > li:nth-child({}) > a > "
                "div.RestaurantReviewItem__ReviewContent > div > p".format(
                    i)).text)
        except NoSuchElementException:  # except no such element error.
            """
            review count가 실시간 변동을 해당 사이트 css 코드에 처리가 안되 있을 경우 
            크롤링이 멈추기 때문에 이 NoSuchElementException 처리 진행.
            """
            pass

        # get review score.
        try:
            review_score.append(browser.find_element_by_xpath(
                "/ html / body / main / article / div[1] / div[1] / div / "
                "section[3] / ul / li[{}] / a / div[3] / span".format(i)).text)
        except InvalidSelectorException:
            pass

        reviews.append(review_score)

    return reviews


def get_information(link):
    """
    Get data of the input link.
    -> (name, loc, category, review_counts, good_counts, well_counts, bad_counts, reviews.)

    :param link: target keyword links to scrape review data.
    :return: List
    """
    browser = load_chrome_browser()
    browser.get(link)

    # get target scraping data.
    name = browser.find_element_by_class_name("restaurant_name").text
    loc = browser.find_element_by_class_name("only-desktop").text
    category = browser.find_element_by_css_selector(
        "section.restaurant-detail > table >"
        " tbody > tr:nth-child(3) > td > span").text
    review_counts = int(browser.find_element_by_class_name(
        "RestaurantReviewList__AllCount").text)
    good_counts = int(browser.find_element_by_css_selector(
        "section.RestaurantReviewList > header > "
        "ul > li:nth-child(2) > button > span").text)
    well_counts = int(browser.find_element_by_css_selector(
        "section.RestaurantReviewList > header > "
        "ul > li:nth-child(3) > button > span").text)
    bad_counts = int(browser.find_element_by_css_selector(
        "section.RestaurantReviewList > header > "
        "ul > li:nth-child(4) > button > span").text)
    reviews = get_review(link)

    return [name, loc, category, review_counts, good_counts,
            well_counts, bad_counts, reviews]


if __name__ == "__main__":
    # test.
    print(get_review("https://www.mangoplate.com/restaurants/zR60EufZwL"))
    # print(get_information("https://www.mangoplate.com/restaurants/zR60EufZwL")) # mangoplate site.


