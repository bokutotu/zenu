#!/bin/bash

git ls-files '*.json' | while read file; do
    jq -c . "$file" > "$file.tmp" && mv "$file.tmp" "$file"
done

