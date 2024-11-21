#!/usr/bin/env python3
from datetime import datetime, date
from influxdb import InfluxDBClient
#import algorithm as alg
from algorithm_detect import stream_SST
import time, math
import operator
import argparse
import sys, os
import numpy as np
from dateutil import tz
from dateutil.parser import parse
import webbrowser
from utils import local_time_epoch, epoch_time_local, influx_query_time_epoch
from utils import write_influx
import warnings
warnings.filterwarnings("ignore")

duration = 60

def get_args_info(args):

    if (args.start is None and args.end is not None):
        print("Error: Please specify the start time!")
        sys.exit(1)

    src = {'ip': '', 'port': '8086', 'db': '', 'user':'', 'passw':'', 'ssl': True}
    dst = {'ip': '', 'port': '8086', 'db': '', 'user':'', 'passw':'', 'ssl': True}
    src['ip'] = args.src_ip # influxdb IP address
    src['db'] = args.src_db    
    dst['ip'] = args.dst_ip
    dst['db'] = args.dst_db
    if args.src_ip.find('https://') != -1:
        src['ssl'] = True
    else:
        src['ssl'] = False

    if args.dst_ip.find('https://') != -1:
        dst['ssl'] = True
    else:
        dst['ssl'] = False

    src['user'] = dst['user']= args.user
    src['passw'] = dst['passw']= args.passw

    if args.debug == 'True':
        debug = True 
    else:
        debug = False

    if(args.start == None):
        startEpoch = datetime.now().timestamp() - duration
    else:
        print(args.start)
        startEpoch = local_time_epoch(args.start, "America/New_York")
        
    if(args.end == None):
        endEpoch = startEpoch + 20
        endSet = False
    else:
        endEpoch = local_time_epoch(args.end, "America/New_York")
        endSet = True

    url = dst['ip'] + ":3000/d/XkT06J3Sz/sleep-monitoring?var-mac=" + str(args.unit)

    appedix = ''
    if args.append_version == "True":
        appedix = args.version
    
    #back_compatible_names
    # table_data_list = args.table_data_names.split(":")
    # table_data_names = {
    #     'BS': [table_data_list[0]+appedix, table_data_list[1]],
    #     'HR': [table_data_list[2]+appedix, table_data_list[3]],
    #     'RR': [table_data_list[4]+appedix, table_data_list[5]],
    #     'SP': [table_data_list[6]+appedix, table_data_list[7]],
    #     'DP': [table_data_list[8]+appedix, table_data_list[9]],
    #     'SQ': [table_data_list[10]+appedix, table_data_list[11]],
    #     'BM': [table_data_list[12]+appedix, table_data_list[13]]
    # }
    # print (table_data_names)
    table_data_names = []
    table_name = 'sensor1_AC_mag_score'

    url = url + "&var-BedName=" + str(args.unit)
    if(args.start is not None):
        url = url + "&from=" + str(int(startEpoch*1000)) #+ "000"
    else:
        url = url + "&from=now-2m"
    if(args.end is not None):
        url = url + "&to=" + str(int(endEpoch*1000)) #+ "000"
    else:
        url = url + "&to=now"
    url = url + "&orgId=1&refresh=1s"

    return src, dst, table_data_names, startEpoch, endEpoch, endSet, url, debug


########### main entrance ########
def main(args):
  progname = sys.argv[0]
  # if(len(sys.argv)<2):
  #   print("Usage: %s mac [start] [end] [ip] [https/http]" %(progname))
  #   print("Example: %s b8:27:eb:97:f5:ac   # start with current time and run in real-time as if in a node" %(progname))
  #   print("Example: %s b8:27:eb:97:f5:ac 2020-08-13T02:03:00.200 # start with the specified time and run non-stop" %(progname))
  #   print("Example: %s b8:27:eb:97:f5:ac 2020-08-13T02:03:00.200 2020-08-13T02:05:00.030 # start and end with the specified time" %(progname))
  #   print("Example: %s b8:27:eb:97:f5:ac 2020-08-13T02:03:00.200 2020-08-13T02:05:00.030 sensorweb.us https # specify influxdb IP and http/https" %(progname))
  #   quit()

 # Parameters from Config file
  buffersize   = 60 # config.get('general', 'buffersize')
  samplingrate = 4000 # int(config.get('general', 'samplingrate'))

  maxbuffersize   = 10*samplingrate
  buffer      = []
  buffertime  = []
  unit = args.unit
  src, dst, table_data_name, startEpoch, endEpoch, endSet, url, debug = get_args_info(args)
  print(src, dst, table_data_name, startEpoch, endEpoch, endSet, url, debug)
  print("### Start time:", startEpoch, " ### \n")
  print("### End time:", endEpoch, " ### \n")
  print("Click here to see the results in Grafana:\n\n" + url)
  webbrowser.open(url, new=2)
  startEpoch = math.floor(startEpoch)
  epoch2 = startEpoch - 1 # int( (current - datetime(1970,1,1)).total_seconds())
  epoch1 = epoch2 - 1

  thres1=0.2 #(normally, thres2 < thres1) thres1 is threshold for detecting anomalies' starts
  thres2=0.1 #thres2 is threshold for detecting anomalies' starts
  state=0

  results = []

  order=10
  lag=10
  win_length=20

  Score_start=np.zeros(1) # get the initial score, Score_start
  # x1 = np.empty(order, dtype=np.float64) 
  x1 = np.ones(order, dtype=np.float16) 
  x1 = np.random.rand(order)
  x1 = np.round(x1,2)
  print("shape of x1:",x1.shape)
  x1 /= np.linalg.norm(x1)
  score_start, x1 = 1, x1 #detect.SingularSpectrumTransformation(win_length=win_length, x0=x1, n_components=2,order=order, lag=lag,is_scaled=True).score_online(startdata)
  Score_start=score_start+Score_start*10**4
  print("start score:",Score_start)

  try:
    client = InfluxDBClient(src['ip'].split('//')[1], src['port'], src['user'], src['passw'], src['db'], src['ssl'])
  except Exception as e:
    print("main(), DB access error:")
    print("Error", e)
    quit()


    
  while True:
    current = datetime.now().timestamp() #(datetime.utcnow() - datetime(1970,1,1)).total_seconds()
    epoch2 = epoch2 + 1
    epoch1 = epoch1 + 1

    if (endSet == False and (current-epoch2) < 20):
      time.sleep(1)
      if(debug): 
         print("Waiting*********")

    if (endSet and epoch2 > endEpoch):
      if(debug): print("**** Ended as ", epoch2, " > ", endEpoch, " ***")
      print("Click here to see the results in Grafana:\n\n" + url)
      quit()
    print('**************')
    if(debug): print('start:', epoch_time_local(epoch1, "America/New_York"), 'end:', epoch_time_local(epoch2, "America/New_York"))

    query = 'SELECT "value" FROM waveform_Current WHERE ("location" = \''+unit+'\')  and time >= '+ str(int(epoch1*10e8))+' and time <= '+str(int(epoch2*10e8))

    print(query)

    try:
      result = client.query(query)
    except Exception as e:
      print("main(), no data in the query time period:")
      print("Error", e)
      time.sleep(1)
      # quit()

    print(result)
    points = list(result.get_points())
    values =  list(map(operator.itemgetter('value'), points))
    times  =  list(map(operator.itemgetter('time'),  points))
    print(values)
    if len(values) <=0: continue
    for i in range(len(times)):
      times[i] = influx_query_time_epoch(times[i], "UTC")

    # the buffer management modules
    buffertime = buffertime + times
    buffer     = buffer + values
    buffLen    = len(buffer)

    if(debug):
      print("buffLen: ", buffLen)
      if(buffLen>0):
        print("Buffer Time (America/New_York):    " + epoch_time_local(buffertime[0], "America/New_York") 
            + "  -   " + epoch_time_local(buffertime[-1], "America/New_York"))

    # Cutting the buffer when overflow
    if(buffLen > maxbuffersize):
        difSize = buffLen - maxbuffersize
        del buffer[0:difSize]
        del buffertime[0:difSize]
        buffLen    = buffLen - difSize
    # get more data if the buffer does not have enough data for the minimum window size
    # print(buffertime)
    if (buffLen <= 5*win_length): continue
    timestamp = buffertime[-1] #epoch2 # 
    timestamp = math.floor(timestamp)# round(timestamp)
    print('Timestamp here:', timestamp)
    nowtime = epoch_time_local(timestamp, "America/New_York")
    # timestamp = local_time_epoch(nowtime[:-1], "UTC")

    data=buffer[-5*win_length:]

    stream=np.array(data)/1000000000  #### the new data coming through
    stream = np.nan_to_num(stream, nan=0.0)

    print("Shape of stream data: ",stream.shape)
    # print("Stream Data", stream)
    # lastdata=start ### the initial start of the algorithm
    score,dur,x1= stream_SST(stream,win_length,n_component=2,order=order,lag=lag,x0=x1) #,state_last=state,thres1=thres1,thres2=thres2
    score = np.round(score,4)

    if score >= thres1:  #and state_last==0  
      print("the anomaly starts") 
      state=1 
    else:
      state=0


    #print('nowtime:', nowtime)
    print("The anomaly score for current time point is ",score)
    print("The time that processes", dur)
    print("The current state is:", state)
    print('timestamp', timestamp)
    timestamp = int(epoch2*10e8)   #locacl_time_epoch(str(nowtime[:-1]), "UTC")
    

    #############################  CHANGE TO NEW FORMAT
    write_influx(dst, unit, 'sensor1_AC_mag_score', 'score', [score], timestamp, 1)
    write_influx(dst, unit, 'sensor1_AC_mag_score', 'state', [state], timestamp, 1)
    print('Written to timestamp:', timestamp)


if __name__== '__main__':
  parser = argparse.ArgumentParser(description='Node Test - Smart Plugs', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("unit", type=str, help='BDot MAC address', default='04:04:04:04:04:04')
  parser.add_argument("--start", type=str, default='2024-11-06T09:32:10.000', help='start time')
  parser.add_argument("--end", type=str, default='2024-11-06T10:48:30.000', help='end time')
  parser.add_argument("--sample", type=str, default='1000', help='Sampling Rate')    
  parser.add_argument('--src_ip', type=str, default='https://sensordata.engr.uga.edu',
                      help='the default source influxdb server')   
  parser.add_argument('--dst_ip', type=str, default='https://sensordata.engr.uga.edu',
                      help='the default dst influxdb server')     
  parser.add_argument('--src_db', type=str, default='waveform',
                      help='the default source influxdb DB name')       
  parser.add_argument('--dst_db', type=str, default='waveform',
                      help='the default source influxdb DB name')                 
  parser.add_argument('--table_data_names', type=str, default='vitals:occupancy:vitals:heartrate:vitals:respiratoryrate:vitals:systolic:vitals:diastolic:vitals:quality:vitals:movement', 
                      help='the table_data_names of BS, HR, RR, SP, DP, SQ, BM')
  parser.add_argument('--append_version', type=str, default='False', 
                      help='whether the version is appended to table_name or not: True/False')                
  parser.add_argument('--algo_name', type=str, default='SST', 
                      help='the default algorithm name')
  parser.add_argument('--user', type=str, default='smartplug', 
                      help='the default usename')
  parser.add_argument('--passw', type=str, default='joint@122', 
                      help='the default password')
  parser.add_argument('--debug', type=str, default='True', 
                      help='the debug mode: True/False')


  args = parser.parse_args()
  print('Args:', args)

  main(args)