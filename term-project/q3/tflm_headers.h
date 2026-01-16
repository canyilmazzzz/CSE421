#pragma once

// TFLM minimal headers
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// TensorFlow Lite Micro types
typedef enum {
    kTfLiteNoType = 0,
    kTfLiteFloat32 = 1,
    kTfLiteInt32 = 2,
    kTfLiteInt8 = 3,
    kTfLiteInt64 = 4,
    kTfLiteBool = 6,
    kTfLiteInt16 = 7,
} TfLiteType;

// Error codes
typedef enum {
    kTfLiteOk = 0,
    kTfLiteError = 1,
    kTfLiteDelegateError = 2
} TfLiteStatus;

// Tensor structure
typedef struct TfLiteTensor {
    TfLiteType type;
    void* data;
    size_t bytes;
    struct {
        float scale;
        int32_t zero_point;
    } params;
    int32_t* dims;
    size_t dims_size;
} TfLiteTensor;

// Interpreter structure (simplified)
typedef struct TfLiteInterpreter TfLiteInterpreter;

// Model structure
typedef struct TfLiteModel TfLiteModel;

// API functions
#ifdef __cplusplus
extern "C" {
#endif

// Model functions
TfLiteModel* tflite_model_create(const void* model_data, size_t model_size);
void tflite_model_delete(TfLiteModel* model);

// Interpreter functions
TfLiteInterpreter* tflite_interpreter_create(
    const TfLiteModel* model,
    uint8_t* tensor_arena,
    size_t arena_size);
void tflite_interpreter_delete(TfLiteInterpreter* interpreter);
TfLiteStatus tflite_interpreter_allocate_tensors(TfLiteInterpreter* interpreter);
TfLiteStatus tflite_interpreter_invoke(TfLiteInterpreter* interpreter);

// Tensor access
TfLiteTensor* tflite_interpreter_get_input_tensor(
    TfLiteInterpreter* interpreter, int32_t index);
TfLiteTensor* tflite_interpreter_get_output_tensor(
    TfLiteInterpreter* interpreter, int32_t index);

#ifdef __cplusplus
}
#endif