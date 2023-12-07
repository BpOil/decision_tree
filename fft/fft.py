import numpy as np 
import plotly.express as px
import plotly.graph_objects as go 
import pandas as pd 

# tshark -r ./2015-03-17/snort.log.1426550408 -Y "http.request.method == GET && ip.src == 192.168.0.0/24" -T fields -e frame.time -e http.host -e ip.src -E header=y -E separator=, -E quote=d > fft/http_internal.csv
# data came from netresec.com/?page=PcapFiles "Network Forensics"
# Interesting Frequencies 
'''
Frequency = 1 / Period 
0.0167 Hz = 1 cycle/60  sec
0.0083 Hz = 1 cycle/120 sec
0.0033 Hz = 1 cycle/300 sec
0.0017 Hz = 1 cycle/600 sec  
'''

# init variables
file_name = "./http_internal.csv"
period = '5S' #1min or 30S
sample_rate = 1
filterData = False
filterCol = 'url'
filterValues = 'skype.com|google.com|track.adform.net'
filterTime = False
startTime = '09:00'
endTime = '10:00'
displayGraph = True
displayTable = True

# import csv 
orgdata = pd.read_csv(f"{file_name}",index_col='time', parse_dates=True)
# filter out as much "known good" noise as possible (i.e. remove google.com, facebook.com, etc.)
if filterData:
    data = orgdata[not orgdata[filterCol].str.contains(filterValues)]
else:
    data = orgdata
# limit the dataframe to just one column 
data = data[filterCol]
# apply a time filter to the data if desired
if filterTime:
    data = data.between_time(startTime,endTime)

# math stuff
countsperperiod = data.resample(period).count()
print(f"length of countsperperiod: {len(countsperperiod)}")
    # print(f"countsperperiod: {countsperperiod}")
# remove the DC component of the FFT
countsperperiod = countsperperiod - np.mean(countsperperiod)
# calculate the amplitudes 
rfft = np.fft.rfft(countsperperiod)
# calculate the frequencies 
frequencies = np.fft.rfftfreq(len(countsperperiod),d=(1/sample_rate))
abs_rfft = abs(rfft)

# print debug info 
print(f"max freq: {max(frequencies)}")
print(f"max abs rfft: {max(abs_rfft)}")
#print(f"length of rfft: {len(rfft)}")
#print(f"length of frequencies: {len(frequencies)}")
#print(f"rfft: {rfft}")
#print(f"frequencies: {frequencies}")

# put the rfft and rfftfreq arrays together in a DataFrame
graph = pd.DataFrame(data=[frequencies,abs_rfft]).T
graph.columns=['Hz','Amp']

# calculate the std and mean for the amplitude of the frequences 
std_amp = np.std(abs_rfft)
mean_amp = np.mean(abs_rfft)
print(f"standard div of amp: {std_amp}")
print(f"avergae of amp:      {mean_amp}")

# make a dataframe to contain all amplitudes that fall outside X stds of the mean 
outlier_amps = graph.copy()
for amp in abs_rfft:
    if not amp >= (mean_amp + (std_amp * 3)):
        outlier_amps = outlier_amps.drop(outlier_amps[outlier_amps['Amp'] == amp].index[0])
outlier_amps = outlier_amps.sort_values(by=['Amp'])

# display the frequency line graph 
if displayGraph:
    fig = px.line(x=frequencies, y=(abs_rfft))
    fig.show()
# display the outlier amplitude table graph
if displayTable:
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(outlier_amps.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=outlier_amps.transpose().values.tolist(),
                fill_color='lavender',
                align='left'))
    ])

    fig.show()


