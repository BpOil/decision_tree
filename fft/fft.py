import numpy as np 
import plotly.express as px
import pandas as pd 

# tshark -r ./2015-03-17/snort.log.1426550408 -Y "http.request.method == GET && ip.src == 192.168.0.0/24" -T fields -e frame.time -e http.host -e ip.src -E header=y -E separator=, -E quote=d > fft/http_internal.csv

# data came from netresec.com/?page=PcapFiles "Network Forensics"

file_name = "./http_internal.csv"
period = '10S' #1min or 30S
sample_rate = 1
filterData = False
filterCol = 'url'
filterValues = 'skype.com|google.com|track.adform.net'
filterTime = False
startTime = '09:00'
endTime = '10:00'

orgdata = pd.read_csv(f"{file_name}",index_col='time', parse_dates=True)
if filterData:
    data = orgdata[orgdata[filterCol].str.contains(filterValues) != True]
data = data['url']
if filterTime:
    data = data.between_time(startTime,endTime)
countsperperiod = data.resample(period).count()
print(f"length of countsperperiod: {len(countsperperiod)}")
print(f"countsperperiod: {countsperperiod}")

rfft = np.fft.rfft(countsperperiod)
frequencies = np.fft.rfftfreq(len(countsperperiod),d=(1/sample_rate))
#print(f"length of rfft: {len(rfft)}")
#print(f"length of frequencies: {len(frequencies)}")
#print(f"rfft: {rfft}")
#print(f"frequencies: {frequencies}")

fig = px.line(x=frequencies, y=(abs(rfft)))
fig.show()


