#!/bin/bash
if [ "$#" -ne 6 ]; then
  echo "Usage: ./rm_influx_data.sh homedots.us algtest vitals b8:27:eb:b3:d7:88 2023-01-08T19:54:38 2023-01-08T19:54:38"
  exit 1
fi
# ./rm_influx_data.sh sensorweb.us algtest 2021-02-05 2021-02-06
server=$1 #sensorweb.us
dbname=$2 #algtest
tbname=$3
location=$4
start=$5 #2021-02-05
end=$6 #2021-02-06
# Convert from RFC3339 to UNIX timestamp in nanoseconds
unameOut="$(uname -s)"
echo $unameOut

case "${unameOut}" in
    "Linux")     
      start=$(date -d "$start" "+%s") && end=$(date -d "$end" "+%s") 
      ;;
    "Darwin")    
      start=$(date -j -f "%Y-%m-%dT%H:%M:%S" "$start" "+%s") && end=$(date -j -f "%Y-%m-%dT%H:%M:%S" "$end" "+%s") 
      ;;
    *)          
      echo "UNKNOWN:${unameOut}" && exit
      ;;
esac
# start=$(date -d "$start" "+%s")
# end=$(date -d "$end" "+%s")
# Add 9 zeros to the end of the timestamp
echo $start $end
start="${start}000000000"
end="${end}000000000"
echo $start $end

# curl -k -POST "https://$server:8086/query?pretty=true&db=$dbname" -u beddot:HDots2020 --data-urlencode "q=DELETE FROM "$tbname" WHERE time >= $start and time <= $end and "location" = '$location' "
curl -k -POST "https://$server:8086/query?pretty=true&db=$dbname" -u smartplug:joint@122 --data-urlencode "q=DELETE FROM "$tbname" WHERE time >= $start and time <= $end and "location" = '$location' "
echo "DELETE FROM "$tbname" WHERE time >= $start and time <= $end and "location" = '$location' "
#and vitalsigns,location=b8:27:eb:b3:d7:88
# curl -k -G "https://sensorweb.us:8086/query?pretty=true&db=algtest" -u test:sensorweb --data-urlencode "q=DELETE WHERE time >= '2021-02-05' and time <= '2021-02-06'"
