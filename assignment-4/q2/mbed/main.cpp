#include "mbed.h"

// Model data
#include "kws_mlp_model.h"

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

using namespace tflite;

// Error reporter
static MicroErrorReporter micro_error_reporter;
ErrorReporter* error_reporter = &micro_error_reporter;

// Tensor arena (adjust if AllocateTensors fails)
constexpr int kTensorArenaSize = 40 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

// Resolver (all ops – simple & safe)
static AllOpsResolver resolver;

// Interpreter
static MicroInterpreter* interpreter;

// Tensors
static TfLiteTensor* input;
static TfLiteTensor* output;

int main() {
    printf("STM32 TFLite Micro start\n");

    // Load model
    const Model* model = GetModel(kws_mlp_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema mismatch!\n");
        while (1);
    }

    // Create interpreter
    static MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate memory
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors() failed\n");
        while (1);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    printf("Model loaded successfully\n");

    /*
     * Your model:
     * Dense(1, sigmoid)
     * input shape: [1, 2]
     */

    while (true) {
        // Example input: x = 1.0, y = -1.0
        input->data.f[0] = 1.0f;
        input->data.f[1] = -1.0f;

        if (interpreter->Invoke() != kTfLiteOk) {
            printf("Invoke failed\n");
            continue;
        }

        float result = output->data.f[0];
        printf("Model output: %f\n", result);

        ThisThread::sleep_for(1000ms);
    }
}
