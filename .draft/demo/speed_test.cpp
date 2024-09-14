#include <stdio.h>
#include <time.h>

int main() {
    long long int count = 1333278019;  // 循环次数
    long long int a = 13, b = 17, c = 11, d = -4;  // 初始变量
    long long int checksum = 0;  // 汇总变量，用于防止优化
    clock_t start, end;
    double time_taken;

    // 记录开始时间
    start = clock();

    for(long long int i = 0; i < count; i++) {
        // 执行四则运算，但不改变原始变量的值
        long long int temp1 = a + b;
        long long int temp2 = b * c;
        long long int temp3 = c - d;
        // 为了避免除以零的情况，确保a不为零
        long long int temp4 = (a != 0) ? (d / a) : 0;

        // 将运算结果累加到checksum中
        checksum += temp1 + temp2 + temp3 + temp4;
    }

    // 记录结束时间
    end = clock();

    // 计算耗时（秒）
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    // 输出结果
    printf("Time taken: %f seconds\n", time_taken);
    printf("Checksum: %lld\n", checksum);  // 打印checksum以确保运算未被优化掉

    return 0;
}
