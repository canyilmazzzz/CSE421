#include "mbed.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

extern const unsigned char kws_perceptron_tflite[];
extern const unsigned int kws_perceptron_tflite_len;

constexpr int kTensorArenaSize = 8 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

int main() {
    printf("TFLM init...\n");

    const tflite::Model* model = tflite::GetModel(kws_perceptron_tflite);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema mismatch!\n");
        return 1;
    }

    // Only ONE Dense layer → only need FullyConnected
    static tflite::MicroMutableOpResolver<1> resolver;
    resolver.AddFullyConnected();

    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                                kTensorArenaSize);

    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printf("AllocateTensors failed\n");
        return 1;
    }

    TfLiteTensor* input = interpreter.input(0);

    // Example input (replace with MFCC / Hu Moments)
    for (int i = 0; i < input->dims->data[1]; i++) {
        input->data.f[i] = 0.5f;
    }

    if (interpreter.Invoke() != kTfLiteOk) {
        printf("Invoke failed\n");
        return 1;
    }

    TfLiteTensor* output = interpreter.output(0);
    float prediction = output->data.f[0];

    printf("Prediction: %f\n", prediction);

    while (true) {
        thread_sleep_for(1000);
    }
}
