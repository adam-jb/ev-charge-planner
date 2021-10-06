

import requests
import re
from selenium import webdriver
from selenium.webdriver import ActionChains




###ope website driver
poly_url='https://polymarket.com/market/will-100-million-people-have-received-a-dose-of-an-approved-covid-19-vaccine-in-the-us-by-april-1-2021'
driver=webdriver.Firefox()
driver.get(poly_url)
time.sleep(1)


###click button by element name
driver.find_elements_by_class_name("LoginButton_button__6kq8X")[0].click()

# click into textbook and add text
element=driver.find_element_by_xpath("/html/body/div/div/main/form/input")
ActionChains(driver).move_to_element(element).click(element).send_keys('add my text input here!').perform()

# click on element using ActionChains (mouse) rather than element by class name
element=driver.find_element_by_xpath("/html/body/div/div/main/form/button")
ActionChains(driver).move_to_element(element).click(element).perform()

