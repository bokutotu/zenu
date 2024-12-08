cudnn 9以上が必要っぽい

# build
```bash
mkdir build
cd build
cmake ..
cmake --build . -j32
```

# debug
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
```
