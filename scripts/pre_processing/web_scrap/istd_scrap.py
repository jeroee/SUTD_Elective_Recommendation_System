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

driver.get('https://istd.sutd.edu.sg/education/undergraduate/course-catalogue/')

# links = driver.find_element_by_xpath("//div[@id='activity_filter_chosen']")'
# //*[@id="blog-1-post-6150"]/div/div[1]/ul[1]/li/div/a'
# //*[@id="blog-1-post-6153"]/div/div[1]/ul[1]/li/div/a

# link = driver.find_element_by_tag_name('a')
# print(link)
links = driver.find_elements_by_css_selector('article > div > div.fusion-flexslider.flexslider.fusion-post-slideshow > ul.slides > li > div > a')
count=0
courses = [] 
for link in links:
    link_name = link.get_attribute('href')
    # print(link_name)
    courses.append(link_name)
    count+=1
print(f'number of courses:{count}')

istd = 'https://istd.sutd.edu.sg'
epd = 'https://epd.sutd.edu.sg'
esd = 'https://esd.sutd.edu.sg'
istd_courses = [course for course in courses if istd in course]
epd_courses = [course for course in courses if epd in course]
esd_courses = [course for course in courses if esd in course]
# print(istd_courses)
# print(esd_courses)
# print(epd_courses)
# print(f'number of istd courses: {len(istd_courses)}')
# print(f'number of esd courses: {len(esd_courses)}')
# print(f'number of epd courses: {len(epd_courses)}')



# settling istd courses

# create table for istd courses
# df = pd.DataFrame(columns =['Name','Course Description, Prerequisites, Learning Objectives, Measurable Outcomes, Topics Covered'])
courses = []
errors = []
for idx,link_name in enumerate(istd_courses):
    course={}
    print('\n')
    print(f'navigating to {link_name}')
    driver.get(link_name)
    time.sleep(0.5)
    '''
    do the scraping within each website
    '''
    print(f'course count {idx+1}')
    course_name = driver.find_element_by_class_name('entry-title')
    print(f'course_name:{course_name.text}')
    course['name'] = course_name.text
    try:
        course_description = driver.find_element_by_xpath("//h4[contains(.,'Course Description')]/following-sibling::p")
        print(f'course description: {course_description.text}')
        print('\n') 
        course['description'] = course_description.text
    except Exception as e:
        errors.append((course_name.text,e))
        print(f'no course description')
        course['description'] = None
    try:
        pre_requisites = driver.find_element_by_xpath("//h4[contains(.,'requisite')]/following-sibling::ol \
            | //h4[contains(.,'requisite')]/following-sibling::ul") # finding folowing sibling with any tag name
        print(f'pre-requsities: {pre_requisites.text}')
        course['pre_requisite'] = [pre_requisites.text.split('\n')]
    except Exception as e:
        errors.append((course_name.text,e))
        print(f'no pre-requisite')
        course['pre_requisite'] = []
    print('\n')
    try: 
        learning_objectives = driver.find_element_by_xpath("//h4[contains(.,'Learning Objective')]/following-sibling::ol \
            | //h4[contains(.,'Learning Objective')]/following-sibling::ul")  # finding folowing sibling with any tag name
        print(f'learning objectives:{learning_objectives.text}')
        course['learning_objectives'] = [learning_objectives.text.split('\n')]
    except Exception as e:
        errors.append((course_name.text,e))
        print(f'no learning objectives')
        course['learning_objectives'] = []
    print('\n')
    try: 
        measureable_outcomes = driver.find_element_by_xpath("//h4[contains(.,'Measurable Outcome')]/following-sibling::ol \
            | //h4[contains(.,'Measurable Outcome')]/following-sibling::ul")
        print(f'measuerable outcomes:{measureable_outcomes.text}')
        course['measurable_outcomes'] = [measureable_outcomes.text.split('\n')]
    except Exception as e:
        errors.append((course_name.text,e))
        print('no measurable outcomes')
        course['measurable_outcomes'] = []
    print('\n') 
    try:
        topics_covered = driver.find_element_by_xpath("//h4[contains(.,'Topics Covered')]/following-sibling::ol \
            | //h4[contains(.,'Topics Covered')]/following-sibling::ul\
            | //h5[contains(.,'Topics Covered')]/following-sibling::ol \
            | //h5[contains(.,'Topics Covered')]/following-sibling::ul")
        print(f'topics covered:{topics_covered.text}')
        course['topics_covered'] = [topics_covered.text.split('\n')]
    except Exception as e:
        errors.append((course_name.text,e))
        print('no topics covered')
        course['topics_covered'] = []
    print('\n')          
    time.sleep(0.5)
    courses.append(course)

# print(type(courses))

# with open('istd_courses.json','w') as file:
#     json.dump(courses, file)

# print('Errors: ')
# for i in errors:
#     print(i)
driver.quit()


# # settling esd courses
# for link_name in esd_courses:
#     print(f'navigating to {link_name}')
#     driver.get(link_name)
#     time.sleep(0.5)
#     '''
#     do the scraping within each website
#     '''
#     driver.back()
#     time.sleep(0.5)

# # settling epd courses
# for link_name in epd_courses:
#     print(f'navigating to {link_name}')
#     driver.get(link_name)
#     time.sleep(0.5)
#     '''
#     do the scraping within each website
#     '''
#     driver.back()
#     time.sleep(0.5)
