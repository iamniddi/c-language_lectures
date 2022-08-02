#include <stdio.h>
#include <stdlib.h> 
#include <time.h>

#define DEPTH 3     //뉴런의 층 갯수(깊이)
#define ROW 5       //input(output)의 갯수
#define COL 6       //ROW값 +1 

double*** weight;
double** gradient;
double** input;
double** sigma;

double outputTarget[COL] = { 1, 3, 3, 3, 3, 3 }; //outputTarget[0]은 1또는 0으로 고정(bias위치)

double alpha = -1 * 0.000125;
double bias = 1;

static double ReLU(double x)
{
    if (x > 0) {
        return x;
    }
    return 0;
}

static double DeActivateReLU(double x)
{
    if (x > 0) {
        return 1;
    }
    return 0;
}

double getActivation(double x) {
    return x;
}

double getDifferentialActivation(double x) {
    return 1;
}

double matrixMultiplication(double**** arr1, double*** arr2, int num1, int num2) {
    double sum = 0;
    for (int a = 0; a < COL; a++) {
        sum = sum + (*arr1)[num2][a][num1 - 1] * (*arr2)[a][num2];
    }
    return sum;
}

double transposeMatrixMultiplication(double*** arr1, double**** arr2, int num1, int num2, int num3) {
    double sum = 0;
    for (int a = 0; a < COL - 1; a++) {
        sum = sum + (*arr1)[a][num1] * (*arr2)[num3][num2][a];
    }

    return sum;
}

void getGradient() {
    for (int a = 0; a < COL - 1; a++) {
        gradient[a][DEPTH - 1] = (input[a + 1][DEPTH] - outputTarget[a + 1]);
    }

    for (int i = DEPTH - 2; i >= 0; i--) {
        for (int a = 0; a < COL - 1; a++) {
            gradient[a][i] = transposeMatrixMultiplication(&gradient, &weight, i + 1, a + 1, i + 1);
        }
    }

    return;
}

void backPropagation() {
    for (int i = DEPTH - 1; i >= 0; i--) {
        for (int a = COL - 1; a >= 0; a--) {
            for (int q = ROW - 1; q >= 0; q--) {
                weight[i][a][q] = weight[i][a][q] + alpha * gradient[q][i] * input[a][i];
                getGradient();
            }
        }
    }
}

void feedForward() {
    for (int i = 1; i < DEPTH + 1; i++) {
        input[0][i] = bias;
        for (int a = 1; a < COL; a++) {
            sigma[a][i] = matrixMultiplication(&weight, &input, a, i - 1);
            input[a][i] = getActivation(sigma[a][i]);
        }
    }
}

void randomWeight() {
    srand((unsigned int)time(NULL));
    for (int i = 0; i < DEPTH; i++)
    {
        for (int j = 0; j < COL; j++)
        {
            for (int q = 0; q < ROW; q++)
            {
                weight[i][j][q] = rand() / (double)RAND_MAX * 0.1;
            }
        }
    }
}

void weightPrint() {
    for (int i = 0; i < DEPTH; i++)
    {
        for (int j = 0; j < COL; j++)
        {
            for (int q = 0; q < ROW; q++)
            {
                printf("%f ", weight[i][j][q]);
            }
            printf("\n");
        }
        printf("\n");
    }

    return;
}

void gradientPrint() {
    for (int j = 0; j < COL - 1; j++)
    {
        for (int q = 0; q < DEPTH; q++)
        {
            printf("%f ", gradient[j][q]);
        }
        printf("\n");
    }
    return;
}

void outputPrint() {
    for (int a = 0; a < COL; a++) {
        printf("%f ", input[a][DEPTH]);
    }
    return;
}


int main()
{
    double count = 1;

    weight = malloc(sizeof(double**) * DEPTH);                  //여기서부터

    for (int i = 0; i < DEPTH; i++)
    {
        weight[i] = malloc(sizeof(double*) * COL);

        for (int j = 0; j < COL; j++)
        {
            weight[i][j] = malloc(sizeof(double) * ROW);
        }
    }

    gradient = malloc(sizeof(double*) * (COL - 1));

    for (int i = 0; i < (COL - 1); i++)
    {
        gradient[i] = malloc(sizeof(double) * DEPTH);
    }

    input = malloc(sizeof(double*) * COL);

    for (int i = 0; i < COL; i++)
    {
        input[i] = malloc(sizeof(double) * (DEPTH + 1));
    }

    sigma = malloc(sizeof(double*) * COL);

    for (int i = 0; i < COL; i++)
    {
        sigma[i] = malloc(sizeof(double) * (DEPTH + 1));
    }                                                           //여기까지 동적할당 배열 생성 코드
    
    for (int a = 0; a < COL; a++) { //입력층 초기화
        input[a][0] = 1;
    }

    randomWeight();         //랜덤한 가중치 입력
    feedForward();          //입력값으로 신경망 통과 후 output 출력

    getGradient();//graident 계산

    while (input[1][DEPTH]-3 * input[1][0]<0.0000001) {
        count = count + 0.1;
        input[0][0] = 1;
        printf("%f ", count);
        for (int i = 1; i < COL; i++) {
            input[i][0] = count;
        }

        for (int i = 1; i < COL; i++) {
            outputTarget[i] = count * 3;
        }

        feedForward();
        backPropagation();
        for (int a = 0; a < COL; a++) {
            printf("%f ", input[a][DEPTH]);
        }
        printf("\n");           //while 문으로 1차함수 학습 (y=3x)
    }

    scanf_s("%lf", &count);     //cmd에 인풋값 입력
    for (int i = 1; i < 6; i++) {
        input[i][0] = count;
    }

    feedForward();

    for (int a = 0; a < COL; a++) {
        printf("%f ", input[a][DEPTH]);     //학습한 신경망에 인풋값 통과 후 아웃풋 출력
    }

    
    for (int i = 0; i < DEPTH; i++)  //여기서부터 동적할당 메모리 해제 코드
    {

        for (int j = 0; j < COL; j++)
        {
            free(weight[i][j]);
        }
        free(weight[i]);
    }

    for (int i = 0; i < (COL - 1); i++)
    {
        free(gradient[i]);
    }

    for (int i = 0; i < (COL - 1); i++)
    {
        free(input[i]);
    }

    for (int i = 0; i < (COL - 1); i++)
    {
        free(sigma[i]);
    }

    return 0;
}