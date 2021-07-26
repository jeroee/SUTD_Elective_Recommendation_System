import json

input_file_istd = '../../../data/web_scrap/istd_courses.json'
input_file_esd = '../../../data/web_scrap/esd_courses.json'
output_filed_merged = '../../../data/web_scrap/course_info.json'
# open json file
with open(input_file_istd) as f:
  istd_courses = json.load(f)

with open(input_file_esd) as f:
  esd_courses = json.load(f)

merged_courses =  istd_courses + esd_courses

istd_electives = ['50.006 User Interface Design and Implementation',
       '50.007 Machine Learning', '50.012 Networks',
       '50.017 Graphics and Visualisation', '50.020 Network Security',
       '50.021 Artificial Intelligence',
       '50.033 Foundations of Game Design and Development',
       '50.035 Computer Vision',
       '50.036 Foundations of Distributed Autonomous Systems',
       '50.037 Blockchain Technology', '50.038 Computational Data Science',
       '50.039 Theory and Practice of Deep Learning',
       '50.040 Natural Language Processing',
       '50.041 Distributed Systems and Computing',
       '50.042 Foundations of Cybersecurity',
       '50.043 Database Systems',
       '50.044 System Security', '50.045 Information Retrieval',
       '50.046 Cloud Computing and Internet of Things',
       '50.047 Mobile Robotics', '50.048 Computational Fabrication',
       'Service Design Studio', '01.116 AI for Healthcare (Term 7)',
       '01.117 Brain-Inspired Computing and its Applications (Term 8)',
       '01.102 Energy Systems and Management',
       '01.104 Networked Life', 
       '01.107 Urban Transportation']

esd_electives = ['40.230 Sustainable Engineering',
       '40.232 Water Resources Management',
       '40.240 Investment Science',
       '40.242 Derivative Pricing and Risk Management',
       '40.260 Supply Chain Management',
       '40.302 Advanced Topics in Optimisation#',
       '40.305 Advanced Topics in Stochastic Modelling#',
       '40.316 Game Theory', '40.317 Financial Systems Design',
       '40.318 Supply Chain Digitalisation and Design',
       '40.319 Statistical and Machine Learning',
       '40.320 Airport Systems Planning and Design',
       '40.321 Airport Systems Modelling and Simulation',
       '40.323 Equity Valuation', 
       '40.324 Fundamentals of Investing']

electives = istd_electives + esd_electives
elective_courses = []

for i in merged_courses:
  name = i['name']
  if name in electives:
    elective_courses.append(i)

print(f'total number of courses: {len(elective_courses)}')
with open(output_filed_merged,'w') as file:
    json.dump(elective_courses, file) # comment out this if want to include the cores

    #json.dump(elective_courses, file) # uncomment if want to include the cores