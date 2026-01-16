#include "tflm_core.h"
#include <cstring>
#include <cstdlib>

// Basit memory allocation
struct SimpleAllocator {
    uint8_t* arena;
    size_t arena_size;
    size_t used;
};

static SimpleAllocator allocator = {nullptr, 0, 0};

void* tflm_allocate(size_t size) {
    if (allocator.arena && allocator.used + size <= allocator.arena_size) {
        void* ptr = allocator.arena + allocator.used;
        allocator.used += size;
        return ptr;
    }
    return malloc(size);
}

void tflm_free(void* ptr) {
    if (!allocator.arena || ptr < allocator.arena || 
        ptr >= allocator.arena + allocator.arena_size) {
        free(ptr);
    }
}

// Basit interpreter implementasyonu
struct SimpleInterpreter {
    const uint8_t* model_data;
    TFLM_Tensor input_tensor;
    TFLM_Tensor output_tensor;
    bool allocated;
};

TFLM_Interpreter tflm_create_interpreter(const uint8_t* model_data) {
    SimpleInterpreter* interpreter = (SimpleInterpreter*)malloc(sizeof(SimpleInterpreter));
    if (!interpreter) return nullptr;
    
    memset(interpreter, 0, sizeof(SimpleInterpreter));
    interpreter->model_data = model_data;
    
    // Basit tensor setup
    // NOT: Bu değerleri modelinden parse etmelisin!
    interpreter->input_tensor.type = 3; // kTfLiteInt8 = 3
    interpreter->input_tensor.scale = 0.003921568859368563f; // 1/255
    interpreter->input_tensor.zero_point = -128;
    interpreter->input_tensor.bytes = 25; // NN_DIM
    
    interpreter->output_tensor.type = 3; // kTfLiteInt8
    interpreter->output_tensor.scale = 0.003921568859368563f;
    interpreter->output_tensor.zero_point = -128;
    interpreter->output_tensor.bytes = 1;
    
    return (TFLM_Interpreter)interpreter;
}

bool tflm_allocate_tensors(TFLM_Interpreter interpreter_ptr) {
    SimpleInterpreter* interpreter = (SimpleInterpreter*)interpreter_ptr;
    if (!interpreter) return false;
    
    // Input için memory allocate et
    interpreter->input_tensor.data = malloc(interpreter->input_tensor.bytes);
    if (!interpreter->input_tensor.data) return false;
    
    // Output için memory allocate et
    interpreter->output_tensor.data = malloc(interpreter->output_tensor.bytes);
    if (!interpreter->output_tensor.data) {
        free(interpreter->input_tensor.data);
        return false;
    }
    
    interpreter->allocated = true;
    return true;
}

bool tflm_invoke(TFLM_Interpreter interpreter_ptr) {
    SimpleInterpreter* interpreter = (SimpleInterpreter*)interpreter_ptr;
    if (!interpreter || !interpreter->allocated) return false;
    
    // DEMO: Basit çıkarım
    // Void pointer'ları uygun tipe cast et
    uint8_t* output_ptr = (uint8_t*)interpreter->output_tensor.data;
    uint8_t* input_ptr = (uint8_t*)interpreter->input_tensor.data;
    
    // Kopyala (demo amaçlı)
    size_t copy_size = (interpreter->output_tensor.bytes < interpreter->input_tensor.bytes) 
                       ? interpreter->output_tensor.bytes 
                       : interpreter->input_tensor.bytes;
    
    for (size_t i = 0; i < copy_size; i++) {
        output_ptr[i] = input_ptr[i];
    }
    
    return true;
}

TFLM_Tensor* tflm_get_input_tensor(TFLM_Interpreter interpreter_ptr, int index) {
    if (index != 0) return nullptr;
    SimpleInterpreter* interpreter = (SimpleInterpreter*)interpreter_ptr;
    return interpreter ? &interpreter->input_tensor : nullptr;
}

TFLM_Tensor* tflm_get_output_tensor(TFLM_Interpreter interpreter_ptr, int index) {
    if (index != 0) return nullptr;
    SimpleInterpreter* interpreter = (SimpleInterpreter*)interpreter_ptr;
    return interpreter ? &interpreter->output_tensor : nullptr;
}

void tflm_free_interpreter(TFLM_Interpreter interpreter_ptr) {
    SimpleInterpreter* interpreter = (SimpleInterpreter*)interpreter_ptr;
    if (interpreter) {
        if (interpreter->allocated) {
            free(interpreter->input_tensor.data);
            free(interpreter->output_tensor.data);
        }
        free(interpreter);
    }
}