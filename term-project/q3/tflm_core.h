#pragma once

// TFLM minimal core - FlatBuffers bağımlılığı olmayan
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Model yapısı
typedef struct {
    const uint8_t* data;
    size_t size;
} TFLM_ModelBuffer;

// Tensor yapısı
typedef struct {
    void* data;
    int32_t* dims;
    int32_t type;
    float scale;
    int32_t zero_point;
    int32_t bytes;
} TFLM_Tensor;

// Interpreter yapısı
typedef void* TFLM_Interpreter;

// API fonksiyonları
TFLM_Interpreter tflm_create_interpreter(const uint8_t* model_data);
bool tflm_allocate_tensors(TFLM_Interpreter interpreter);
bool tflm_invoke(TFLM_Interpreter interpreter);
TFLM_Tensor* tflm_get_input_tensor(TFLM_Interpreter interpreter, int index);
TFLM_Tensor* tflm_get_output_tensor(TFLM_Interpreter interpreter, int index);
void tflm_free_interpreter(TFLM_Interpreter interpreter);

#ifdef __cplusplus
}
#endif