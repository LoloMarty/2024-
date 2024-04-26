#include <stdio.h>
#include <time.h>

int main()
{
    int intResult;

    double doubleResult;
    //long double doubleResult;

    clock_t start, end;
    double cpu_time_used1;
    double cpu_time_used2;
    double timeDifference;
    double timePercentage;
    int iterations = 100000000; // 100000000
    int loopBound = 4;

    for (int loops = 0; loops < loopBound; loops++)
    {
        printf("\n\nIteration #%d\nUsing Number: %d\n", loops + 1, iterations);

        // Measure execution time for integer arithmetic
        start = clock();
        intResult = 1;
        for (int i = 1; i < iterations; i++)
        {
            intResult += i;
            intResult *= i;
        }

        end = clock();
        cpu_time_used1 = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Integer arithmetic execution time: %f seconds\n", cpu_time_used1);

        // Measure execution time for double arithmetic
        start = clock();
        doubleResult = 1.0;
        for (int i = 1; i < iterations; i++)
        {
            doubleResult += (double)i;
            doubleResult *= (double)i;
        }
        end = clock();
        cpu_time_used2 = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Double arithmetic execution time: %f seconds\n", cpu_time_used2);

        timeDifference = (cpu_time_used1 > cpu_time_used2) ? cpu_time_used1 - cpu_time_used2 : cpu_time_used2 - cpu_time_used1;
        timePercentage = (cpu_time_used1 > cpu_time_used2) ? ((cpu_time_used1 / cpu_time_used2)-1)*100 : ((cpu_time_used2 / cpu_time_used1)-1)*100;
        printf("Time Difference (s): %f\n", timeDifference);
        printf("Time Difference (%%): %f\n", timePercentage);

        iterations *= 10;
    }

    scanf("");

    return 0;
}
