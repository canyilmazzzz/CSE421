#include "mbed.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

extern const unsigned char hdr_perceptron_model[];
extern const unsigned int hdr_perceptron_model_len;

constexpr int kTensorArenaSize = 4 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

int main() {
    printf("HDR Perceptron TFLM Demo\n");

    const tflite::Model* model = tflite::GetModel(hdr_perceptron_model);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema mismatch!\n");
        return 1;
    }

    static tflite::MicroMutableOpResolver<2> resolver;
    resolver.AddFullyConnected();
    resolver.AddLogistic();  // sigmoid

    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                                kTensorArenaSize);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("Tensor allocation failed\n");
        return 1;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    // Model expects 7 Hu moments (float32)
    float example_input[7] = {0.1f, -0.2f, 0.05f, 0.3f, -0.1f, 0.02f, 0.15f};

    for (int i = 0; i < 7; i++) {
        input->data.f[i] = example_input[i];
    }

    if (interpreter.Invoke() != kTfLiteOk) {
        printf("Invoke failed\n");
        return 1;
    }

    float prediction = output->data.f[0];
    printf("Prediction: %f\n", prediction);

    while (true) {
        thread_sleep_for(1000);
    }
}
