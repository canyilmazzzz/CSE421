#include "tflm_headers.h"
#include <cstdlib>
#include <cstring>

// Simple implementation - we'll replace this with real TFLM later
struct SimpleModel {
    const uint8_t* data;
    size_t size;
};

struct SimpleInterpreter {
    SimpleModel* model;
    TfLiteTensor input_tensor;
    TfLiteTensor output_tensor;
    uint8_t* arena;
    size_t arena_size;
};

// Model functions
TfLiteModel* tflite_model_create(const void* model_data, size_t model_size) {
    SimpleModel* model = (SimpleModel*)malloc(sizeof(SimpleModel));
    if (!model) return nullptr;
    
    model->data = (const uint8_t*)model_data;
    model->size = model_size;
    
    return (TfLiteModel*)model;
}

void tflite_model_delete(TfLiteModel* model_ptr) {
    SimpleModel* model = (SimpleModel*)model_ptr;
    free(model);
}

// Interpreter functions
TfLiteInterpreter* tflite_interpreter_create(
    const TfLiteModel* model_ptr,
    uint8_t* tensor_arena,
    size_t arena_size) {
    
    const SimpleModel* model = (const SimpleModel*)model_ptr;
    SimpleInterpreter* interpreter = (SimpleInterpreter*)malloc(sizeof(SimpleInterpreter));
    if (!interpreter) return nullptr;
    
    // Initialize tensors based on your model
    // From your Python code: input shape (1, 25), output shape (1, 1), int8 quantized
    
    // Input tensor
    interpreter->input_tensor.type = kTfLiteInt8;
    interpreter->input_tensor.params.scale = 0.003921568859368563f; // 1/255
    interpreter->input_tensor.params.zero_point = -128;
    interpreter->input_tensor.bytes = 25; // 1 * 25 * 1 (int8)
    interpreter->input_tensor.dims = (int32_t*)malloc(2 * sizeof(int32_t));
    interpreter->input_tensor.dims[0] = 1;  // batch
    interpreter->input_tensor.dims[1] = 25; // features
    interpreter->input_tensor.dims_size = 2;
    
    // Output tensor
    interpreter->output_tensor.type = kTfLiteInt8;
    interpreter->output_tensor.params.scale = 0.003921568859368563f;
    interpreter->output_tensor.params.zero_point = -128;
    interpreter->output_tensor.bytes = 1; // 1 * 1 * 1 (int8)
    interpreter->output_tensor.dims = (int32_t*)malloc(2 * sizeof(int32_t));
    interpreter->output_tensor.dims[0] = 1; // batch
    interpreter->output_tensor.dims[1] = 1; // single output
    interpreter->output_tensor.dims_size = 2;
    
    // Use arena if provided
    if (tensor_arena && arena_size >= 1024) {
        interpreter->arena = tensor_arena;
        interpreter->arena_size = arena_size;
        interpreter->input_tensor.data = tensor_arena;
        interpreter->output_tensor.data = tensor_arena + 25; // After input
    } else {
        interpreter->arena = nullptr;
        interpreter->arena_size = 0;
        interpreter->input_tensor.data = malloc(25);
        interpreter->output_tensor.data = malloc(1);
    }
    
    interpreter->model = (SimpleModel*)model;
    
    return (TfLiteInterpreter*)interpreter;
}

void tflite_interpreter_delete(TfLiteInterpreter* interpreter_ptr) {
    SimpleInterpreter* interpreter = (SimpleInterpreter*)interpreter_ptr;
    if (interpreter) {
        free(interpreter->input_tensor.dims);
        free(interpreter->output_tensor.dims);
        
        if (!interpreter->arena) {
            free(interpreter->input_tensor.data);
            free(interpreter->output_tensor.data);
        }
        
        free(interpreter);
    }
}

TfLiteStatus tflite_interpreter_allocate_tensors(TfLiteInterpreter* interpreter_ptr) {
    // Already allocated in create
    return kTfLiteOk;
}

TfLiteStatus tflite_interpreter_invoke(TfLiteInterpreter* interpreter_ptr) {
    SimpleInterpreter* interpreter = (SimpleInterpreter*)interpreter_ptr;
    
    // DEMO: Simple "inference" - average of first few inputs
    int8_t* input = (int8_t*)interpreter->input_tensor.data;
    int8_t* output = (int8_t*)interpreter->output_tensor.data;
    
    // Simple calculation (replace with real TFLM later)
    int32_t sum = 0;
    for (int i = 0; i < 5 && i < 25; i++) {
        sum += input[i];
    }
    output[0] = (int8_t)(sum / 5);
    
    return kTfLiteOk;
}

// Tensor access
TfLiteTensor* tflite_interpreter_get_input_tensor(
    TfLiteInterpreter* interpreter_ptr, int32_t index) {
    if (index != 0) return nullptr;
    SimpleInterpreter* interpreter = (SimpleInterpreter*)interpreter_ptr;
    return &interpreter->input_tensor;
}

TfLiteTensor* tflite_interpreter_get_output_tensor(
    TfLiteInterpreter* interpreter_ptr, int32_t index) {
    if (index != 0) return nullptr;
    SimpleInterpreter* interpreter = (SimpleInterpreter*)interpreter_ptr;
    return &interpreter->output_tensor;
}