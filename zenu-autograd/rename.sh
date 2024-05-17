#!/bin/bash

# get args "from", "to"
from=$1
to=$2
# error handling 
if [ -z "$from" ] || [ -z "$to" ]; then
  echo "Usage: $0 <from> <to>"
  exit 1
fi

# Loop over every .rs file in the current directory and its subdirectories
for file in $(find src/functions -name "*.rs")
do
  sed -i "s/$from/$to/g" "$file"
done
