#include "mbed.h"

// Model
#include "temperature_pred_model.h"

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
constexpr int kTensorArenaSize = 60 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

// Resolver
static AllOpsResolver resolver;

// Interpreter
static MicroInterpreter* interpreter;

// Tensors
static TfLiteTensor* input;
static TfLiteTensor* output;

int main() {
    printf("STM32F746NG Temperature Prediction Model\n");

    // Load model
    const Model* model = GetModel(temperature_pred_mlp_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema mismatch!\n");
        while (true);
    }

    // Create interpreter
    static MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors failed\n");
        while (true);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    printf("Model ready\n");

    /*
     * Expected model:
     * Input:  float32[1][N]  (N = number of previous temperatures)
     * Output: float32[1][1]  (predicted temperature)
     */

    while (true) {
        // Example input (replace with real temperature history)
        input->data.f[0] = 22.1f;
        input->data.f[1] = 22.3f;
        input->data.f[2] = 22.4f;
        input->data.f[3] = 22.6f;
        input->data.f[4] = 22.7f;

        if (interpreter->Invoke() != kTfLiteOk) {
            printf("Invoke failed\n");
            continue;
        }

        float predicted_temp = output->data.f[0];
        printf("Predicted temperature: %.2f\n", predicted_temp);

        ThisThread::sleep_for(2000ms);
    }
}
