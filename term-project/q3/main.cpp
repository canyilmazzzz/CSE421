#include "mbed.h"
#include "model_data.h"
#include "sf_params.h"
#include "lib/tflite_micro/tflm_headers.h"
#include <cstdio>
#include <cstring>
#include <cmath>

static BufferedSerial pc(USBTX, USBRX, 115200);

// Tensor arena - increased for real TFLM
static constexpr int kTensorArenaSize = 24 * 1024; // 24KB
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// Helper functions
void serial_print(const char* str) {
    pc.write(str, strlen(str));
}

// Min-Max scaling
static inline float minmax_scale(float x, float mn, float sc) {
    return (x - mn) * sc;
}

static inline float minmax_unscale(float y_scaled) {
    // y = y_scaled/scale + min
    return (y_scaled / Y_SCALE[0]) + Y_MIN[0];
}

// Ridge regression predict (scaled space)
static float ridge_predict_scaled(const float x_lr_scaled[10]) {
    float y = RIDGE_B;
    for (int i = 0; i < 10; i++) {
        y += RIDGE_W[i] * x_lr_scaled[i];
    }
    return y;
}

int main() {
    serial_print("=== SmartFarming TFLM Fusion ===\r\n");
    
    // 1. Create model
    serial_print("1. Creating TFLite model...\r\n");
    TfLiteModel* model = tflite_model_create(g_model, g_model_len);
    if (!model) {
        serial_print("ERROR: Failed to create model\r\n");
        return 1;
    }
    
    // 2. Create interpreter
    serial_print("2. Creating interpreter...\r\n");
    TfLiteInterpreter* interpreter = tflite_interpreter_create(
        model, tensor_arena, kTensorArenaSize);
    if (!interpreter) {
        serial_print("ERROR: Failed to create interpreter\r\n");
        tflite_model_delete(model);
        return 1;
    }
    
    // 3. Allocate tensors
    serial_print("3. Allocating tensors...\r\n");
    if (tflite_interpreter_allocate_tensors(interpreter) != kTfLiteOk) {
        serial_print("ERROR: Failed to allocate tensors\r\n");
        tflite_interpreter_delete(interpreter);
        tflite_model_delete(model);
        return 1;
    }
    
    // 4. Get input/output tensors
    serial_print("4. Getting tensors...\r\n");
    TfLiteTensor* input_tensor = tflite_interpreter_get_input_tensor(interpreter, 0);
    TfLiteTensor* output_tensor = tflite_interpreter_get_output_tensor(interpreter, 0);
    
    if (!input_tensor || !output_tensor) {
        serial_print("ERROR: Failed to get tensors\r\n");
        tflite_interpreter_delete(interpreter);
        tflite_model_delete(model);
        return 1;
    }
    
    // Print tensor info
    char info[128];
    snprintf(info, sizeof(info), 
        "Input: type=%d, bytes=%d, scale=%.6f, zp=%d\r\n",
        input_tensor->type, input_tensor->bytes,
        input_tensor->params.scale, input_tensor->params.zero_point);
    serial_print(info);
    
    snprintf(info, sizeof(info),
        "Output: type=%d, bytes=%d, scale=%.6f, zp=%d\r\n",
        output_tensor->type, output_tensor->bytes,
        output_tensor->params.scale, output_tensor->params.zero_point);
    serial_print(info);
    
    // Demo data
    float x_lr_raw[10] = {
        35.0f,  6.5f, 28.0f, 120.0f, 60.0f,
        7.0f, 120.0f, 0.65f, 4.0f, 110.0f
    };
    
    serial_print("\n5. Starting main loop...\r\n");
    serial_print("==========================\r\n");
    
    int iteration = 0;
    const float ALPHA = 0.5f; // Fusion parameter
    
    while (true) {
        iteration++;
        
        // === 1. Prepare LR features ===
        float x_lr_scaled[10];
        for (int i = 0; i < 10; i++) {
            x_lr_scaled[i] = minmax_scale(x_lr_raw[i], LR_MIN[i], LR_SCALE[i]);
            // Clamp to [0, 1]
            if (x_lr_scaled[i] < 0) x_lr_scaled[i] = 0;
            if (x_lr_scaled[i] > 1) x_lr_scaled[i] = 1;
        }
        
        // Ridge prediction (in scaled space)
        float y_ridge_scaled = ridge_predict_scaled(x_lr_scaled);
        
        // === 2. Prepare NN input ===
        // We need to create 25 features for NN
        // For demo, we'll use: 10 LR features + 15 synthetic features
        if (input_tensor->type == kTfLiteInt8) {
            int8_t* input_data = (int8_t*)input_tensor->data;
            
            // First 10: scaled LR features (quantized)
            for (int i = 0; i < 10 && i < 25; i++) {
                float scaled = minmax_scale(x_lr_raw[i], NN_MIN[i], NN_SCALE[i]);
                // Quantize: q = round(x / scale) + zero_point
                int32_t q = (int32_t)lrintf(scaled / input_tensor->params.scale) 
                           + input_tensor->params.zero_point;
                // Clamp to int8 range
                if (q < -128) q = -128;
                if (q > 127) q = 127;
                input_data[i] = (int8_t)q;
            }
            
            // Remaining 15: synthetic data (demo)
            for (int i = 10; i < 25; i++) {
                // Simple pattern for demo
                input_data[i] = (int8_t)((iteration + i) % 100 - 50);
            }
        }
        
        // === 3. Run inference ===
        TfLiteStatus invoke_status = tflite_interpreter_invoke(interpreter);
        
        if (invoke_status != kTfLiteOk) {
            serial_print("ERROR: Inference failed\r\n");
            ThisThread::sleep_for(2000ms);
            continue;
        }
        
        // === 4. Get NN output ===
        float y_nn_scaled = 0.0f;
        if (output_tensor->type == kTfLiteInt8 && output_tensor->bytes >= 1) {
            int8_t yq = ((int8_t*)output_tensor->data)[0];
            // Dequantize: y = (q - zp) * scale
            y_nn_scaled = ((float)yq - output_tensor->params.zero_point) 
                         * output_tensor->params.scale;
        }
        
        // === 5. Fusion ===
        float y_fused_scaled = ALPHA * y_nn_scaled + (1.0f - ALPHA) * y_ridge_scaled;
        
        // === 6. Unscale to real value ===
        float y_real = minmax_unscale(y_fused_scaled);
        
        // === 7. Print results ===
        char result[256];
        // Simple integer printing for now
        snprintf(result, sizeof(result),
            "Iter %3d | Ridge: %4d | NN: %4d | Yield: %6.0f kg/ha\r\n",
            iteration,
            (int)(y_ridge_scaled * 1000),
            (int)(y_nn_scaled * 1000),
            y_real);
        serial_print(result);
        
        // === 8. Update demo data ===
        x_lr_raw[0] += 0.5f;   // moisture
        x_lr_raw[2] += 0.1f;   // temperature
        x_lr_raw[4] += 0.3f;   // humidity
        x_lr_raw[9] += 1.0f;   // day of year
        
        // Clamp values to reasonable ranges
        if (x_lr_raw[0] > 60.0f) x_lr_raw[0] = 30.0f;
        if (x_lr_raw[2] > 35.0f) x_lr_raw[2] = 20.0f;
        if (x_lr_raw[9] > 365.0f) x_lr_raw[9] = 1.0f;
        
        // === 9. Wait ===
        ThisThread::sleep_for(3000ms);
        
        // Print separator every 5 iterations
        if (iteration % 5 == 0) {
            serial_print("--------------------------\r\n");
        }
    }
    
    // Cleanup (never reached in infinite loop)
    tflite_interpreter_delete(interpreter);
    tflite_model_delete(model);
    
    return 0;
}