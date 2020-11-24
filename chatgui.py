import requests
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


senForTheMatching = ['covid cases in Maharashtra', 'covid cases in Andhra Pradesh', 'covid cases in Karnataka', 'covid cases in Tamil Nadu', 'covid cases in Uttar Pradesh', 'covid cases in Kerala', 'covid cases in Delhi', 'covid cases in West Bengal', 'covid cases in Odisha', 'covid cases in Telangana', 'covid cases in Bihar', 'covid cases in Assam', 'covid cases in Rajasthan', 'covid cases in Chhattisgarh', 'covid cases in Gujarat', 'covid cases in Madhya Pradesh', 'covid cases in Haryana', 'covid cases in Punjab', 'covid cases in Jharkhand', 'covid cases in Jammu and Kashmir', 'covid cases in Uttarakhand', 'covid cases in Goa', 'covid cases in Puducherry', 'covid cases in Tripura', 'covid cases in Himachal Pradesh', 'covid cases in Manipur', 'covid cases in Arunachal Pradesh', 'covid cases in Chandigarh', 'covid cases in Meghalaya', 'covid cases in Nagaland', 'covid cases in Ladakh', 'covid cases in Andaman and Nicobar Islands', 'covid cases in Sikkim', 'covid cases in Mizoram', 'covid cases in Daman and Diu', 'covid cases in Dadra and Nagar Haveli', 'covid cases in Lakshadweep']
senForDist = ['Andaman and Nicobar Islands:Nicobars', 'Andaman and Nicobar Islands:North and Middle Andaman', 'Andaman and Nicobar Islands:South Andaman', 'Andaman and Nicobar Islands:Unknown', 'Andhra Pradesh:Foreign Evacuees', 'Andhra Pradesh:Anantapur', 'Andhra Pradesh:Chittoor', 'Andhra Pradesh:East Godavari', 'Andhra Pradesh:Guntur', 'Andhra Pradesh:Krishna', 'Andhra Pradesh:Kurnool', 'Andhra Pradesh:Other State', 'Andhra Pradesh:Prakasam', 'Andhra Pradesh:S.P.S. Nellore', 'Andhra Pradesh:Srikakulam', 'Andhra Pradesh:Visakhapatnam', 'Andhra Pradesh:Vizianagaram', 'Andhra Pradesh:West Godavari', 'Andhra Pradesh:Y.S.R. Kadapa', 'Arunachal Pradesh:Anjaw', 'Arunachal Pradesh:Changlang', 'Arunachal Pradesh:East Kameng', 'Arunachal Pradesh:East Siang', 'Arunachal Pradesh:Kamle', 'Arunachal Pradesh:Kra Daadi', 'Arunachal Pradesh:Kurung Kumey', 'Arunachal Pradesh:Lepa Rada', 'Arunachal Pradesh:Lohit', 'Arunachal Pradesh:Longding', 'Arunachal Pradesh:Lower Dibang Valley', 'Arunachal Pradesh:Lower Siang', 'Arunachal Pradesh:Lower Subansiri', 'Arunachal Pradesh:Namsai', 'Arunachal Pradesh:Pakke Kessang', 'Arunachal Pradesh:Papum Pare', 'Arunachal Pradesh:Shi Yomi', 'Arunachal Pradesh:Siang', 'Arunachal Pradesh:Tawang', 'Arunachal Pradesh:Tirap', 'Arunachal Pradesh:Upper Dibang Valley', 'Arunachal Pradesh:Upper Siang', 'Arunachal Pradesh:Upper Subansiri', 'Arunachal Pradesh:West Kameng', 'Arunachal Pradesh:West Siang', 'Assam:Airport Quarantine', 'Assam:Baksa', 'Assam:Barpeta', 'Assam:Biswanath', 'Assam:Bongaigaon', 'Assam:Cachar', 'Assam:Charaideo', 'Assam:Chirang', 'Assam:Darrang', 'Assam:Dhemaji', 'Assam:Dhubri', 'Assam:Dibrugarh', 'Assam:Dima Hasao', 'Assam:Goalpara', 'Assam:Golaghat', 'Assam:Hailakandi', 'Assam:Hojai', 'Assam:Jorhat', 'Assam:Kamrup', 'Assam:Kamrup Metropolitan', 'Assam:Karbi Anglong', 'Assam:Karimganj', 'Assam:Kokrajhar', 'Assam:Lakhimpur', 'Assam:Majuli', 'Assam:Morigaon', 'Assam:Nagaon', 'Assam:Nalbari', 'Assam:Other State', 'Assam:Sivasagar', 'Assam:Sonitpur', 'Assam:South Salmara Mankachar', 'Assam:Tinsukia', 'Assam:Udalguri', 'Assam:West Karbi Anglong', 'Assam:Unknown', 'Bihar:Araria', 'Bihar:Arwal', 'Bihar:Aurangabad', 'Bihar:Banka', 'Bihar:Begusarai', 'Bihar:Bhagalpur', 'Bihar:Bhojpur', 'Bihar:Buxar', 'Bihar:Darbhanga', 'Bihar:East Champaran', 'Bihar:Gaya', 'Bihar:Gopalganj', 'Bihar:Jamui', 'Bihar:Jehanabad', 'Bihar:Kaimur', 'Bihar:Katihar', 'Bihar:Khagaria', 'Bihar:Kishanganj', 'Bihar:Lakhisarai', 'Bihar:Madhepura', 'Bihar:Madhubani', 'Bihar:Munger', 'Bihar:Muzaffarpur', 'Bihar:Nalanda', 'Bihar:Nawada', 'Bihar:Patna', 'Bihar:Purnia', 'Bihar:Rohtas', 'Bihar:Saharsa', 'Bihar:Samastipur', 'Bihar:Saran', 'Bihar:Sheikhpura', 'Bihar:Sheohar', 'Bihar:Sitamarhi', 'Bihar:Siwan', 'Bihar:Supaul', 'Bihar:Vaishali', 'Bihar:West Champaran', 'Chandigarh:Chandigarh', 'Chhattisgarh:Other State', 'Chhattisgarh:Balod', 'Chhattisgarh:Baloda Bazar', 'Chhattisgarh:Balrampur', 'Chhattisgarh:Bametara', 'Chhattisgarh:Bastar', 'Chhattisgarh:Bijapur', 'Chhattisgarh:Bilaspur', 'Chhattisgarh:Dakshin Bastar Dantewada', 'Chhattisgarh:Dhamtari', 'Chhattisgarh:Durg', 'Chhattisgarh:Gariaband', 'Chhattisgarh:Janjgir Champa', 'Chhattisgarh:Jashpur', 'Chhattisgarh:Kabeerdham', 'Chhattisgarh:Kondagaon', 'Chhattisgarh:Korba', 'Chhattisgarh:Koriya', 'Chhattisgarh:Mahasamund', 'Chhattisgarh:Mungeli', 'Chhattisgarh:Narayanpur', 'Chhattisgarh:Raigarh', 'Chhattisgarh:Raipur', 'Chhattisgarh:Rajnandgaon', 'Chhattisgarh:Sukma', 'Chhattisgarh:Surajpur', 'Chhattisgarh:Surguja', 'Chhattisgarh:Uttar Bastar Kanker', 'Chhattisgarh:Gaurela Pendra Marwahi', 'Delhi:Central Delhi', 'Delhi:East Delhi', 'Delhi:New Delhi', 'Delhi:North Delhi', 'Delhi:North East Delhi', 'Delhi:North West Delhi', 'Delhi:Shahdara', 'Delhi:South Delhi', 'Delhi:South East Delhi', 'Delhi:South West Delhi', 'Delhi:West Delhi', 'Delhi:Unknown', 'Dadra and Nagar Haveli and Daman and Diu:Other State', 'Dadra and Nagar Haveli and Daman and Diu:Dadra and Nagar Haveli', 'Dadra and Nagar Haveli and Daman and Diu:Daman', 'Dadra and Nagar Haveli and Daman and Diu:Diu', 'Goa:Other State', 'Goa:North Goa', 'Goa:South Goa', 'Goa:Unknown', 'Gujarat:Other State', 'Gujarat:Ahmedabad', 'Gujarat:Amreli', 'Gujarat:Anand', 'Gujarat:Aravalli', 'Gujarat:Banaskantha', 'Gujarat:Bharuch', 'Gujarat:Bhavnagar', 'Gujarat:Botad', 'Gujarat:Chhota Udaipur', 'Gujarat:Dahod', 'Gujarat:Dang', 'Gujarat:Devbhumi Dwarka', 'Gujarat:Gandhinagar', 'Gujarat:Gir Somnath', 'Gujarat:Jamnagar', 'Gujarat:Junagadh', 'Gujarat:Kheda', 'Gujarat:Kutch', 'Gujarat:Mahisagar', 'Gujarat:Mehsana', 'Gujarat:Morbi', 'Gujarat:Narmada', 'Gujarat:Navsari', 'Gujarat:Panchmahal', 'Gujarat:Patan', 'Gujarat:Porbandar', 'Gujarat:Rajkot', 'Gujarat:Sabarkantha', 'Gujarat:Surat', 'Gujarat:Surendranagar', 'Gujarat:Tapi', 'Gujarat:Vadodara', 'Gujarat:Valsad', 'Himachal Pradesh:Bilaspur', 'Himachal Pradesh:Chamba', 'Himachal Pradesh:Hamirpur', 'Himachal Pradesh:Kangra', 'Himachal Pradesh:Kinnaur', 'Himachal Pradesh:Kullu', 'Himachal Pradesh:Lahaul and Spiti', 'Himachal Pradesh:Mandi', 'Himachal Pradesh:Shimla', 'Himachal Pradesh:Sirmaur', 'Himachal Pradesh:Solan', 'Himachal Pradesh:Una', 'Haryana:Foreign Evacuees', 'Haryana:Ambala', 'Haryana:Bhiwani', 'Haryana:Charkhi Dadri', 'Haryana:Faridabad', 'Haryana:Fatehabad', 'Haryana:Gurugram', 'Haryana:Hisar', 'Haryana:Italians', 'Haryana:Jhajjar', 'Haryana:Jind', 'Haryana:Kaithal', 'Haryana:Karnal', 'Haryana:Kurukshetra', 'Haryana:Mahendragarh', 'Haryana:Nuh', 'Haryana:Palwal', 'Haryana:Panchkula', 'Haryana:Panipat', 'Haryana:Rewari', 'Haryana:Rohtak', 'Haryana:Sirsa', 'Haryana:Sonipat', 'Haryana:Yamunanagar', 'Jharkhand:Bokaro', 'Jharkhand:Chatra', 'Jharkhand:Deoghar', 'Jharkhand:Dhanbad', 'Jharkhand:Dumka', 'Jharkhand:East Singhbhum', 'Jharkhand:Garhwa', 'Jharkhand:Giridih', 'Jharkhand:Godda', 'Jharkhand:Gumla', 'Jharkhand:Hazaribagh', 'Jharkhand:Jamtara', 'Jharkhand:Khunti', 'Jharkhand:Koderma', 'Jharkhand:Latehar', 'Jharkhand:Lohardaga', 'Jharkhand:Pakur', 'Jharkhand:Palamu', 'Jharkhand:Ramgarh', 'Jharkhand:Ranchi', 'Jharkhand:Sahibganj', 'Jharkhand:Saraikela-Kharsawan', 'Jharkhand:Simdega', 'Jharkhand:West Singhbhum', 'Jammu and Kashmir:Anantnag', 'Jammu and Kashmir:Bandipora', 'Jammu and Kashmir:Baramulla', 'Jammu and Kashmir:Budgam', 'Jammu and Kashmir:Doda', 'Jammu and Kashmir:Ganderbal', 'Jammu and Kashmir:Jammu', 'Jammu and Kashmir:Kathua', 'Jammu and Kashmir:Kishtwar', 'Jammu and Kashmir:Kulgam', 'Jammu and Kashmir:Kupwara', 'Jammu and Kashmir:Mirpur', 'Jammu and Kashmir:Muzaffarabad', 'Jammu and Kashmir:Pulwama', 'Jammu and Kashmir:Punch', 'Jammu and Kashmir:Rajouri', 'Jammu and Kashmir:Ramban', 'Jammu and Kashmir:Reasi', 'Jammu and Kashmir:Samba', 'Jammu and Kashmir:Shopiyan', 'Jammu and Kashmir:Srinagar', 'Jammu and Kashmir:Udhampur', 'Karnataka:Bagalkote', 'Karnataka:Ballari', 'Karnataka:Belagavi', 'Karnataka:Bengaluru Rural', 'Karnataka:Bengaluru Urban', 'Karnataka:Bidar', 'Karnataka:Chamarajanagara', 'Karnataka:Chikkaballapura', 'Karnataka:Chikkamagaluru', 'Karnataka:Chitradurga', 'Karnataka:Dakshina Kannada', 'Karnataka:Davanagere', 'Karnataka:Dharwad', 'Karnataka:Gadag', 'Karnataka:Hassan', 'Karnataka:Haveri', 'Karnataka:Kalaburagi', 'Karnataka:Kodagu', 'Karnataka:Kolar', 'Karnataka:Koppal', 'Karnataka:Mandya', 'Karnataka:Mysuru', 'Karnataka:Other State', 'Karnataka:Raichur', 'Karnataka:Ramanagara', 'Karnataka:Shivamogga', 'Karnataka:Tumakuru', 'Karnataka:Udupi', 'Karnataka:Uttara Kannada', 'Karnataka:Vijayapura', 'Karnataka:Yadgir', 'Kerala:Other State', 'Kerala:Alappuzha', 'Kerala:Ernakulam', 'Kerala:Idukki', 'Kerala:Kannur', 'Kerala:Kasaragod', 'Kerala:Kollam', 'Kerala:Kottayam', 'Kerala:Kozhikode', 'Kerala:Malappuram', 'Kerala:Palakkad', 'Kerala:Pathanamthitta', 'Kerala:Thiruvananthapuram', 'Kerala:Thrissur', 'Kerala:Wayanad', 'Ladakh:Kargil', 'Ladakh:Leh', 'Lakshadweep:Lakshadweep', 'Maharashtra:Ahmednagar', 'Maharashtra:Akola', 'Maharashtra:Amravati', 'Maharashtra:Aurangabad', 'Maharashtra:Beed', 'Maharashtra:Bhandara', 'Maharashtra:Buldhana', 'Maharashtra:Chandrapur', 'Maharashtra:Dhule', 'Maharashtra:Gadchiroli', 'Maharashtra:Gondia', 'Maharashtra:Hingoli', 'Maharashtra:Jalgaon', 'Maharashtra:Jalna', 'Maharashtra:Kolhapur', 'Maharashtra:Latur', 'Maharashtra:Mumbai', 'Maharashtra:Mumbai Suburban', 'Maharashtra:Nagpur', 'Maharashtra:Nanded', 'Maharashtra:Nandurbar', 'Maharashtra:Nashik', 'Maharashtra:Osmanabad', 'Maharashtra:Other State', 'Maharashtra:Palghar', 'Maharashtra:Parbhani', 'Maharashtra:Pune', 'Maharashtra:Raigad', 'Maharashtra:Ratnagiri', 'Maharashtra:Sangli', 'Maharashtra:Satara', 'Maharashtra:Sindhudurg', 'Maharashtra:Solapur', 'Maharashtra:Thane', 'Maharashtra:Wardha', 'Maharashtra:Washim', 'Maharashtra:Yavatmal', 'Meghalaya:East Garo Hills', 'Meghalaya:East Jaintia Hills', 'Meghalaya:East Khasi Hills', 'Meghalaya:North Garo Hills', 'Meghalaya:Ribhoi', 'Meghalaya:South Garo Hills', 'Meghalaya:South West Garo Hills', 'Meghalaya:South West Khasi Hills', 'Meghalaya:West Garo Hills', 'Meghalaya:West Jaintia Hills', 'Meghalaya:West Khasi Hills', 'Manipur:CAPF Personnel', 'Manipur:Bishnupur', 'Manipur:Chandel', 'Manipur:Churachandpur', 'Manipur:Imphal East', 'Manipur:Imphal West', 'Manipur:Jiribam', 'Manipur:Kakching', 'Manipur:Kamjong', 'Manipur:Kangpokpi', 'Manipur:Noney', 'Manipur:Pherzawl', 'Manipur:Senapati', 'Manipur:Tamenglong', 'Manipur:Tengnoupal', 'Manipur:Thoubal', 'Manipur:Ukhrul', 'Manipur:Unknown', 'Madhya Pradesh:Agar Malwa', 'Madhya Pradesh:Alirajpur', 'Madhya Pradesh:Anuppur', 'Madhya Pradesh:Ashoknagar', 'Madhya Pradesh:Balaghat', 'Madhya Pradesh:Barwani', 'Madhya Pradesh:Betul', 'Madhya Pradesh:Bhind', 'Madhya Pradesh:Bhopal', 'Madhya Pradesh:Burhanpur', 'Madhya Pradesh:Chhatarpur', 'Madhya Pradesh:Chhindwara', 'Madhya Pradesh:Damoh', 'Madhya Pradesh:Datia', 'Madhya Pradesh:Dewas', 'Madhya Pradesh:Dhar', 'Madhya Pradesh:Dindori', 'Madhya Pradesh:Guna', 'Madhya Pradesh:Gwalior', 'Madhya Pradesh:Harda', 'Madhya Pradesh:Hoshangabad', 'Madhya Pradesh:Indore', 'Madhya Pradesh:Jabalpur', 'Madhya Pradesh:Jhabua', 'Madhya Pradesh:Katni', 'Madhya Pradesh:Khandwa', 'Madhya Pradesh:Khargone', 'Madhya Pradesh:Mandla', 'Madhya Pradesh:Mandsaur', 'Madhya Pradesh:Morena', 'Madhya Pradesh:Narsinghpur', 'Madhya Pradesh:Neemuch', 'Madhya Pradesh:Niwari', 'Madhya Pradesh:Other Region', 'Madhya Pradesh:Panna', 'Madhya Pradesh:Raisen', 'Madhya Pradesh:Rajgarh', 'Madhya Pradesh:Ratlam', 'Madhya Pradesh:Rewa', 'Madhya Pradesh:Sagar', 'Madhya Pradesh:Satna', 'Madhya Pradesh:Sehore', 'Madhya Pradesh:Seoni', 'Madhya Pradesh:Shahdol', 'Madhya Pradesh:Shajapur', 'Madhya Pradesh:Sheopur', 'Madhya Pradesh:Shivpuri', 'Madhya Pradesh:Sidhi', 'Madhya Pradesh:Singrauli', 'Madhya Pradesh:Tikamgarh', 'Madhya Pradesh:Ujjain', 'Madhya Pradesh:Umaria', 'Madhya Pradesh:Vidisha', 'Mizoram:Aizawl', 'Mizoram:Champhai', 'Mizoram:Hnahthial', 'Mizoram:Khawzawl', 'Mizoram:Kolasib', 'Mizoram:Lawngtlai', 'Mizoram:Lunglei', 'Mizoram:Mamit', 'Mizoram:Saiha', 'Mizoram:Saitual', 'Mizoram:Serchhip', 'Nagaland:Others', 'Nagaland:Dimapur', 'Nagaland:Kiphire', 'Nagaland:Kohima', 'Nagaland:Longleng', 'Nagaland:Mokokchung', 'Nagaland:Mon', 'Nagaland:Peren', 'Nagaland:Phek', 'Nagaland:Tuensang', 'Nagaland:Wokha', 'Nagaland:Zunheboto', 'Odisha:State Pool', 'Odisha:Others', 'Odisha:Angul', 'Odisha:Balangir', 'Odisha:Balasore', 'Odisha:Bargarh', 'Odisha:Bhadrak', 'Odisha:Boudh', 'Odisha:Cuttack', 'Odisha:Deogarh', 'Odisha:Dhenkanal', 'Odisha:Gajapati', 'Odisha:Ganjam', 'Odisha:Jagatsinghpur', 'Odisha:Jajpur', 'Odisha:Jharsuguda', 'Odisha:Kalahandi', 'Odisha:Kandhamal', 'Odisha:Kendrapara', 'Odisha:Kendujhar', 'Odisha:Khordha', 'Odisha:Koraput', 'Odisha:Malkangiri', 'Odisha:Mayurbhanj', 'Odisha:Nabarangapur', 'Odisha:Nayagarh', 'Odisha:Nuapada', 'Odisha:Puri', 'Odisha:Rayagada', 'Odisha:Sambalpur', 'Odisha:Subarnapur', 'Odisha:Sundargarh', 'Punjab:Amritsar', 'Punjab:Barnala', 'Punjab:Bathinda', 'Punjab:Faridkot', 'Punjab:Fatehgarh Sahib', 'Punjab:Fazilka', 'Punjab:Ferozepur', 'Punjab:Gurdaspur', 'Punjab:Hoshiarpur', 'Punjab:Jalandhar', 'Punjab:Kapurthala', 'Punjab:Ludhiana', 'Punjab:Mansa', 'Punjab:Moga', 'Punjab:Pathankot', 'Punjab:Patiala', 'Punjab:Rupnagar', 'Punjab:S.A.S. Nagar', 'Punjab:Sangrur', 'Punjab:Shahid Bhagat Singh Nagar', 'Punjab:Sri Muktsar Sahib', 'Punjab:Tarn Taran', 'Puducherry:Karaikal', 'Puducherry:Mahe', 'Puducherry:Puducherry', 'Puducherry:Yanam', 'Rajasthan:Ajmer', 'Rajasthan:Alwar', 'Rajasthan:Banswara', 'Rajasthan:Baran', 'Rajasthan:Barmer', 'Rajasthan:Bharatpur', 'Rajasthan:Bhilwara', 'Rajasthan:Bikaner', 'Rajasthan:BSF Camp', 'Rajasthan:Bundi', 'Rajasthan:Chittorgarh', 'Rajasthan:Churu', 'Rajasthan:Dausa', 'Rajasthan:Dholpur', 'Rajasthan:Dungarpur', 'Rajasthan:Evacuees', 'Rajasthan:Ganganagar', 'Rajasthan:Hanumangarh', 'Rajasthan:Italians', 'Rajasthan:Jaipur', 'Rajasthan:Jaisalmer', 'Rajasthan:Jalore', 'Rajasthan:Jhalawar', 'Rajasthan:Jhunjhunu', 'Rajasthan:Jodhpur', 'Rajasthan:Karauli', 'Rajasthan:Kota', 'Rajasthan:Nagaur', 'Rajasthan:Other State', 'Rajasthan:Pali', 'Rajasthan:Pratapgarh', 'Rajasthan:Rajsamand', 'Rajasthan:Sawai Madhopur', 'Rajasthan:Sikar', 'Rajasthan:Sirohi', 'Rajasthan:Tonk', 'Rajasthan:Udaipur', 'Sikkim:East Sikkim', 'Sikkim:North Sikkim', 'Sikkim:South Sikkim', 'Sikkim:West Sikkim', 'Sikkim:Unknown', 'Telangana:Foreign Evacuees', 'Telangana:Other State', 'Telangana:Adilabad', 'Telangana:Bhadradri Kothagudem', 'Telangana:Hyderabad', 'Telangana:Jagtial', 'Telangana:Jangaon', 'Telangana:Jayashankar Bhupalapally', 'Telangana:Jogulamba Gadwal', 'Telangana:Kamareddy', 'Telangana:Karimnagar', 'Telangana:Khammam', 'Telangana:Komaram Bheem', 'Telangana:Mahabubabad', 'Telangana:Mahabubnagar', 'Telangana:Mancherial', 'Telangana:Medak', 'Telangana:Medchal Malkajgiri', 'Telangana:Mulugu', 'Telangana:Nagarkurnool', 'Telangana:Nalgonda', 'Telangana:Narayanpet', 'Telangana:Nirmal', 'Telangana:Nizamabad', 'Telangana:Peddapalli', 'Telangana:Rajanna Sircilla', 'Telangana:Ranga Reddy', 'Telangana:Sangareddy', 'Telangana:Siddipet', 'Telangana:Suryapet', 'Telangana:Vikarabad', 'Telangana:Wanaparthy', 'Telangana:Warangal Rural', 'Telangana:Warangal Urban', 'Telangana:Yadadri Bhuvanagiri', 'Telangana:Unknown', 'Tamil Nadu:Railway Quarantine', 'Tamil Nadu:Airport Quarantine', 'Tamil Nadu:Other State', 'Tamil Nadu:Ariyalur', 'Tamil Nadu:Chengalpattu', 'Tamil Nadu:Chennai', 'Tamil Nadu:Coimbatore', 'Tamil Nadu:Cuddalore', 'Tamil Nadu:Dharmapuri', 'Tamil Nadu:Dindigul', 'Tamil Nadu:Erode', 'Tamil Nadu:Kallakurichi', 'Tamil Nadu:Kancheepuram', 'Tamil Nadu:Kanyakumari', 'Tamil Nadu:Karur', 'Tamil Nadu:Krishnagiri', 'Tamil Nadu:Madurai', 'Tamil Nadu:Nagapattinam', 'Tamil Nadu:Namakkal', 'Tamil Nadu:Nilgiris', 'Tamil Nadu:Perambalur', 'Tamil Nadu:Pudukkottai', 'Tamil Nadu:Ramanathapuram', 'Tamil Nadu:Ranipet', 'Tamil Nadu:Salem', 'Tamil Nadu:Sivaganga', 'Tamil Nadu:Tenkasi', 'Tamil Nadu:Thanjavur', 'Tamil Nadu:Theni', 'Tamil Nadu:Thiruvallur', 'Tamil Nadu:Thiruvarur', 'Tamil Nadu:Thoothukkudi', 'Tamil Nadu:Tiruchirappalli', 'Tamil Nadu:Tirunelveli', 'Tamil Nadu:Tirupathur', 'Tamil Nadu:Tiruppur', 'Tamil Nadu:Tiruvannamalai', 'Tamil Nadu:Vellore', 'Tamil Nadu:Viluppuram', 'Tamil Nadu:Virudhunagar', 'Tripura:Dhalai', 'Tripura:Gomati', 'Tripura:Khowai', 'Tripura:North Tripura', 'Tripura:Sipahijala', 'Tripura:South Tripura', 'Tripura:Unokoti', 'Tripura:West Tripura', 'Uttar Pradesh:Agra', 'Uttar Pradesh:Aligarh', 'Uttar Pradesh:Ambedkar Nagar', 'Uttar Pradesh:Amethi', 'Uttar Pradesh:Amroha', 'Uttar Pradesh:Auraiya', 'Uttar Pradesh:Ayodhya', 'Uttar Pradesh:Azamgarh', 'Uttar Pradesh:Baghpat', 'Uttar Pradesh:Bahraich', 'Uttar Pradesh:Ballia', 'Uttar Pradesh:Balrampur', 'Uttar Pradesh:Banda', 'Uttar Pradesh:Barabanki', 'Uttar Pradesh:Bareilly', 'Uttar Pradesh:Basti', 'Uttar Pradesh:Bhadohi', 'Uttar Pradesh:Bijnor', 'Uttar Pradesh:Budaun', 'Uttar Pradesh:Bulandshahr', 'Uttar Pradesh:Chandauli', 'Uttar Pradesh:Chitrakoot', 'Uttar Pradesh:Deoria', 'Uttar Pradesh:Etah', 'Uttar Pradesh:Etawah', 'Uttar Pradesh:Farrukhabad', 'Uttar Pradesh:Fatehpur', 'Uttar Pradesh:Firozabad', 'Uttar Pradesh:Gautam Buddha Nagar', 'Uttar Pradesh:Ghaziabad', 'Uttar Pradesh:Ghazipur', 'Uttar Pradesh:Gonda', 'Uttar Pradesh:Gorakhpur', 'Uttar Pradesh:Hamirpur', 'Uttar Pradesh:Hapur', 'Uttar Pradesh:Hardoi', 'Uttar Pradesh:Hathras', 'Uttar Pradesh:Jalaun', 'Uttar Pradesh:Jaunpur', 'Uttar Pradesh:Jhansi', 'Uttar Pradesh:Kannauj', 'Uttar Pradesh:Kanpur Dehat', 'Uttar Pradesh:Kanpur Nagar', 'Uttar Pradesh:Kasganj', 'Uttar Pradesh:Kaushambi', 'Uttar Pradesh:Kushinagar', 'Uttar Pradesh:Lakhimpur Kheri', 'Uttar Pradesh:Lalitpur', 'Uttar Pradesh:Lucknow', 'Uttar Pradesh:Maharajganj', 'Uttar Pradesh:Mahoba', 'Uttar Pradesh:Mainpuri', 'Uttar Pradesh:Mathura', 'Uttar Pradesh:Mau', 'Uttar Pradesh:Meerut', 'Uttar Pradesh:Mirzapur', 'Uttar Pradesh:Moradabad', 'Uttar Pradesh:Muzaffarnagar', 'Uttar Pradesh:Pilibhit', 'Uttar Pradesh:Pratapgarh', 'Uttar Pradesh:Prayagraj', 'Uttar Pradesh:Rae Bareli', 'Uttar Pradesh:Rampur', 'Uttar Pradesh:Saharanpur', 'Uttar Pradesh:Sambhal', 'Uttar Pradesh:Sant Kabir Nagar', 'Uttar Pradesh:Shahjahanpur', 'Uttar Pradesh:Shamli', 'Uttar Pradesh:Shrawasti', 'Uttar Pradesh:Siddharthnagar', 'Uttar Pradesh:Sitapur', 'Uttar Pradesh:Sonbhadra', 'Uttar Pradesh:Sultanpur', 'Uttar Pradesh:Unnao', 'Uttar Pradesh:Varanasi', 'Uttarakhand:Almora', 'Uttarakhand:Bageshwar', 'Uttarakhand:Chamoli', 'Uttarakhand:Champawat', 'Uttarakhand:Dehradun', 'Uttarakhand:Haridwar', 'Uttarakhand:Nainital', 'Uttarakhand:Pauri Garhwal', 'Uttarakhand:Pithoragarh', 'Uttarakhand:Rudraprayag', 'Uttarakhand:Tehri Garhwal', 'Uttarakhand:Udham Singh Nagar', 'Uttarakhand:Uttarkashi', 'West Bengal:Alipurduar', 'West Bengal:Bankura', 'West Bengal:Birbhum', 'West Bengal:Cooch Behar', 'West Bengal:Dakshin Dinajpur', 'West Bengal:Darjeeling', 'West Bengal:Hooghly', 'West Bengal:Howrah', 'West Bengal:Jalpaiguri', 'West Bengal:Jhargram', 'West Bengal:Kalimpong', 'West Bengal:Kolkata', 'West Bengal:Malda', 'West Bengal:Murshidabad', 'West Bengal:Nadia', 'West Bengal:North 24 Parganas', 'West Bengal:Other State', 'West Bengal:Paschim Bardhaman', 'West Bengal:Paschim Medinipur', 'West Bengal:Purba Bardhaman', 'West Bengal:Purba Medinipur', 'West Bengal:Purulia', 'West Bengal:South 24 Parganas', 'West Bengal:Uttar Dinajpur']
# def getCovidCases(state):
#     response = requests.get("https://api.covidindiatracker.com/state_data.json").json()
#     data = response

#     cases = {}

#     for res in data:
#         state_code = res['id']
#         state_name = res['state']
#         active = res['active']
#         cases[state_code] = state_name , active


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


def getCovidDataFromState(state):
    response = requests.get("https://api.covidindiatracker.com/state_data.json").json()
    data = response

    states = {}

    cases = {}

    for res in data:
        state_code = res['id']
        state_name = res['state']
        active = res['active']
        cases[state_code] = state_name , active
        states[state_name] = state_code

    return cases[state]

def getState(msg):
    msg = msg.split(" ")
    if len(msg) == 5:
        msg = msg[3]+" "+msg[4]
    elif len(msg) == 7:
        msg = msg[3]+" "+msg[4]+" "+msg[5]+" "+msg[6]
    else:
        msg = msg[-1]
    with open('/home/bhavesh/MY/b/python-project-chatbot/stateList.json') as jsonfile:
        data = json.load(jsonfile)
        if msg in data.keys():
            return(data[msg])


def getDataStateDist(data1):
    response = requests.get('https://api.covid19india.org/state_district_wise.json').json()
    data = response
    data1 = data1.split(':')
    state = data1[0]
    dist = data1[1]
    casesIn = "The Cases In "+state+" "+dist+" Are "+str(data[state]['districtData'][dist]['confirmed'])
    return str(casesIn)

#Creating GUI with tkinter
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)


    if msg in senForTheMatching:
        stateR = getState(msg)
        casesOf = getCovidDataFromState(stateR)
        print(casesOf)
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END,"YOU: " +msg + '\n\n')
        ChatLog.insert(END,"Bot: The Active corona cases in "+str(casesOf[0])+" are "+str(casesOf[1])+'\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
    elif msg in senForDist:
        DistData = getDataStateDist(msg)
        print(DistData)
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END,"YOU: " +msg + '\n\n')
        ChatLog.insert(END,DistData+'\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
    else:
        if msg != '':
            ChatLog.config(state=NORMAL)
            ChatLog.insert(END, "You: " + msg + '\n\n')
            ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        
            res = chatbot_response(msg)
            ChatLog.insert(END, "Bot: " + res + '\n\n')
                
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)
    

base = Tk()
base.title("Q/A System")
base.geometry("600x600")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=577,y=6, height=500)
ChatLog.place(x=6,y=6, height=500, width=575)
EntryBox.place(x=128, y=505, height=90, width=575)
SendButton.place(x=6, y=505, height=90)

base.mainloop()
