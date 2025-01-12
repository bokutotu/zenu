#include <gtest/gtest.h>
// main 関数 (GoogleTest が用意しているマクロ)
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

