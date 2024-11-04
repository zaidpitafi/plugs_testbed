#!/bin/bash
PYTHON=$(which python3)
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
table_data_name="sensor1_AC_mag_score:score:sensor1_AC_mag_score:state"

#input="./nodetest.list"
echo $PWD # line 4 already cd to the current directory where the shell script is
echo $1 $2 $3 $4
ip=$2
#user=$3
#passw=$4
input=$1
if [ -z "$input" ]
then
   echo "usage: live_run.sh live_run.list"
   exit
fi

if [ -z "$ip" ]
then
   ip="https://sensordata.engr.uga.edu"
fi

if [ -z "$user" ]
then
   user="algtest"
   passw="sensorweb711"
fi

extract_domain() {
    # Remove "http://" or "https://" using sed
    cleaned_url=$(echo "$1" | sed -e 's/https\?:\/\///')

    # Split the cleaned URL by "/" and take the first part
    domain=$(echo "$cleaned_url" | cut -d "/" -f 1)

    echo "$domain"
}

cd $SCRIPTPATH
while IFS= read -r line
do
  line_array=($line)
  status=${line_array[0]}
  mode=${line_array[1]}
  mac=${line_array[2]}
  ip=${line_array[3]}
  user=${line_array[4]}
  passw=${line_array[5]}

  if [ -z "$mac" ]
  then
    exit
  fi
  
  if [ "$mode" = "T" ]
  then
    baseParams="--append_version=False --table_data_name=${table_data_name} --dst_db=waveform"
  else
    baseParams="--append_version=False --table_data_name=${table_data_name} --dst_db=waveform"
  fi

  baseParams="$baseParams --src_ip=$ip --dst_ip=$ip --user=${user} --passw=${passw}"

  #print the arguments
  echo
  echo "status: $status"
  echo "mac: $mac"
  echo "ip: $ip"
  echo "user: $user"
  echo "passw: $passw"

  param="$baseParams"


  now=$(date +"%T")
  cd $SCRIPTPATH
  SERVICE="$SCRIPTPATH/algtest.py $mac $param"
  process=$(pgrep -f "$SERVICE")
  process=${process[0]}
  echo "the process ID of $mac is $process"
  if [[ ! -z $process ]]
  then
      echo "$SERVICE is running at $now"
      if [ "$status" = "OFF" ]
      then
        echo "kill -9 $process"
        echo
        kill -9 $process
      fi
  else
      echo "$SERVICE is stopped at $now"
      if [ "$status" = "ON" ]
      then
        echo "$PYTHON $SERVICE"
        echo "$status"
        echo
        # nohup /usr/bin/python3 $SERVICE &
        if [ -n "$start_time" ]
        then
          $RM_INFLUX
        fi
        # echo "cd $SCRIPTPATH"
        cd $SCRIPTPATH
        # nohup $PYTHON $SERVICE &
        $PYTHON $SERVICE
      fi
  fi
done < "$input"