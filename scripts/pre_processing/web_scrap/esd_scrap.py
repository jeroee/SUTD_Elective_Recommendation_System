from logging import error
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pandas as pd
import time
import json
# from tqdm import tqdm
import re
import time
# import geopy

options = Options()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
# options.add_argument('--headless')
driver = webdriver.Chrome("chrome_browser/chromedriver", options=options)

driver.get('https://esd.sutd.edu.sg/academics/undergraduate-programme/courses/')

links = driver.find_elements_by_css_selector('article > div > div.fusion-flexslider.flexslider.fusion-post-slideshow > ul.slides > li > div > a')
count=0
esd_courses = [] 
for link in links:
    link_name = link.get_attribute('href')
    # print(link_name)
    esd_courses.append(link_name)
    count+=1
print(f'number of courses:{count}')

courses = []
errors = []
for idx,link_name in enumerate(esd_courses):
    course={}
    print('\n')
    print(f'navigating to {link_name}')
    driver.get(link_name)
    time.sleep(0.5)
    '''
    do the scraping within each website
    '''
    print(f'course count {idx+1}')
    course_name = driver.find_element_by_class_name('fusion-post-title')
    print(f'course_name:{course_name.text}')
    course['name'] = course_name.text

    course_description = driver.find_element_by_xpath("//*[@class='post-content']/p[1]")
    print(f'course description: {course_description.text}')
    course['description'] = course_description.text
    print('\n') 
    try:
        pre_requisites = driver.find_element_by_xpath("//b[contains(.,'equisite')]/following-sibling::a \
            | //b[contains(.,'equisite')]/../following-sibling::ul \
            | //b[contains(.,'equisite')]/../following-sibling::ol \
            | //strong[contains(.,'equisite')]/following-sibling::a \
            | //strong[contains(.,'equisite')]/../following-sibling::ul \
            | //strong[contains(.,'equisite')]/../following-sibling::ol \
            | //h4[contains(.,'equisite')]/following-sibling::ul \
            | //h4[contains(.,'equisite')]/following-sibling::ol")
        print(f'pre-requisites: {pre_requisites.text}')
        course['pre_requisite'] = [pre_requisites.text.split('\n')]
    except Exception as e:
        print('no pre-requisites')
        errors.append((course_name.text, e))
        course['pre_requisite'] = []
    print('\n')
    learning_objectives = driver.find_element_by_xpath("//h4[contains(.,'Learning Objective')]/following-sibling::ol \
            | //h4[contains(.,'Learning Objective')]/following-sibling::ul")
    print(f'learning objectives:{learning_objectives.text}')
    course['learning_objectives'] = [learning_objectives.text.split('\n')]
    print('\n')
    measurable_outcomes = driver.find_element_by_xpath("//h4[contains(.,'Outcome')]/following-sibling::ol \
            | //h4[contains(.,'Outcome')]/following-sibling::ul")
    print(f'measuerable outcomes:{measurable_outcomes.text}')
    course['measurable_outcomes'] = [measurable_outcomes.text.split('\n')]
    print('\n')
    time.sleep(0.5)
    courses.append(course)
    
with open('esd_courses.json','w') as file:
    json.dump(courses, file)

driver.quit()

print('Errors')
for i in errors:
    print(i)
# //*[@id="post-8918"]/div/p[3]/strong[2]
# #post-8918 > div > p:nth-child(7) > strong:nth-child(3)
# //*[@id="post-3715"]/div/p[4]/b
# #post-3715 > div > p:nth-child(8) > b