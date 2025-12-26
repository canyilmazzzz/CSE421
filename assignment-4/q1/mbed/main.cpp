#include "mbed.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>

using std::memcpy;
using std::memset;
using std::memmove;

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "har_mlp_model_data.h"

constexpr int kTensorArenaSize = 40 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

float feature_vector[10];
float ax[80], ay[80], az[80];

void extract_features() {
    float x_mean = 0, y_mean = 0, z_mean = 0;
    int x_pos = 0, y_pos = 0, z_pos = 0;

    for (int i = 0; i < 80; i++) {
        x_mean += ax[i];
        y_mean += ay[i];
        z_mean += az[i];
        if (ax[i] > 0) x_pos++;
        if (ay[i] > 0) y_pos++;
        if (az[i] > 0) z_pos++;
    }

    x_mean /= 80.0f;
    y_mean /= 80.0f;
    z_mean /= 80.0f;

    float sma = 0;
    for (int i = 0; i < 80; i++)
        sma += fabs(ax[i]) + fabs(ay[i]) + fabs(az[i]);
    sma /= 80.0f;

    feature_vector[0] = x_mean;
    feature_vector[1] = y_mean;
    feature_vector[2] = z_mean;
    feature_vector[3] = x_pos;
    feature_vector[4] = y_pos;
    feature_vector[5] = z_pos;
    feature_vector[6] = 0.1f;
    feature_vector[7] = 0.1f;
    feature_vector[8] = 0.1f;
    feature_vector[9] = sma;
}

int main() {
    printf("STM32F746G-DISCO HAR MLP basladi\n");

    for (int i = 0; i < 80; i++) {
        ax[i] = sinf(i * 0.1f);
        ay[i] = cosf(i * 0.1f);
        az[i] = 0.5f;
    }

    extract_features();

    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    const TfLiteModel* model = tflite::GetModel(har_mlp_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema uyumsuz!\n");
        return 1;
    }

    static tflite::MicroMutableOpResolver<2> resolver;
    resolver.AddFullyConnected();
    resolver.AddSoftmax();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter
    );

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors failed\n");
        return 1;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    for (int i = 0; i < 10; i++)
        input->data.f[i] = feature_vector[i];

    if (interpreter->Invoke() != kTfLiteOk) {
        printf("Invoke failed\n");
        return 1;
    }

    int out_size = output->dims->data[1];
    for (int i = 0; i < out_size; i++)
        printf("Class %d: %f\n", i, output->data.f[i]);

    while (true)
        ThisThread::sleep_for(1000ms);
}
