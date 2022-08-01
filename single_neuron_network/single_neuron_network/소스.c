#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

double input;
double weight;
double bias;
double alpha = -1 * 0.1;
double output;

double Activation(double x) {
	return x;
}

void feedForward(double input) {
	double sigma = weight * input + bias;
	output = Activation(sigma);

	return;
}

void backPropagation(double target, double y) {
	weight = weight + alpha * (y - target);

	return;
}

int main(void) {
	double outputTarget;

	weight = 0.1;
	bias = 1;

	printf("인풋값을 입력해주십시요\n");
	scanf("%lf", &input);

	printf("타겟값을 입력해주십시요\n");
	scanf("%lf", &outputTarget);

	feedForward(input);
	while ((outputTarget - output) > 0.0001) {
		feedForward(input);
		backPropagation(outputTarget, output);
		printf("%f %f\n", input, output);
	}

	return 0;
}