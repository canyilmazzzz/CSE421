#include "mbed.h"

// Model
#include "hdr_mlp_model.h"

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

using namespace tflite;

// Error reporter
static MicroErrorReporter micro_error_reporter;
ErrorReporter* error_reporter = &micro_error_reporter;

// Tensor arena (increase if AllocateTensors fails)
constexpr int kTensorArenaSize = 40 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

// Resolver (safe but larger binary)
static AllOpsResolver resolver;

// Interpreter
static MicroInterpreter* interpreter;

// Tensors
static TfLiteTensor* input;
static TfLiteTensor* output;

int main() {
    printf("STM32F746NG TFLite Micro demo\n");

    // Load model from C array
    const Model* model = GetModel(hdr_mlp_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema version mismatch!\n");
        while (true);
    }

    // Create interpreter
    static MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors() failed\n");
        while (true);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    printf("Model loaded successfully\n");

    /*
     * Model:
     * Dense(1, sigmoid)
     * Input shape: [1, 2]
     * Output shape: [1, 1]
     */

    while (true) {
        // Example input (x, y)
        input->data.f[0] = 1.0f;
        input->data.f[1] = -1.0f;

        if (interpreter->Invoke() != kTfLiteOk) {
            printf("Invoke failed\n");
            continue;
        }

        float y = output->data.f[0];
        printf("Output: %f\n", y);

        ThisThread::sleep_for(1000ms);
    }
}
