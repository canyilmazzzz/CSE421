#include "mbed.h"
#include "rtos/ThisThread.h"
#include <cstdint>
#include <cstdio>
#include <cstring>

// ===== TensorFlow Lite Micro =====
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// ===== Model data (har_model_data.cc) =====
extern const unsigned char g_har_model[];
extern const unsigned int  g_har_model_len;

// ===== Serial =====
static BufferedSerial pc(USBTX, USBRX, 115200);

static size_t cstr_len(const char* s) {
    size_t n = 0;
    while (s[n] != '\0') {
        n++;
    }
    return n;
}


static void print_str(const char* s) {
    pc.write(s, cstr_len(s));
}


static void print_float(const char* label, float v) {
    char buf[96];
    int n = snprintf(buf, sizeof(buf), "%s%.6f\r\n", label, v);
    if (n > 0) pc.write(buf, n);
}

// ===== Tensor arena =====
// Başlangıç için 20KB iyi. AllocateTensors fail olursa 24/32KB denersin.
constexpr int kTensorArenaSize = 20 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

int main() {
    print_str("\r\n--- HAR Single Neuron (TFLite Micro) ---\r\n");

    // ===== Load model =====
    const tflite::Model* model = tflite::GetModel(g_har_model);
    if (!model) {
        print_str("ERROR: GetModel() returned null\r\n");
        while (true) ThisThread::sleep_for(1s);
    }

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        print_str("ERROR: Model schema mismatch!\r\n");
        while (true) ThisThread::sleep_for(1s);
    }

    // ===== Resolver (sadece gerekli ops) =====
    // Modelin tipik olarak: Quantize -> FullyConnected -> Dequantize
    static tflite::MicroMutableOpResolver<3> resolver;
    if (resolver.AddQuantize() != kTfLiteOk)       { print_str("AddQuantize failed\r\n"); while (true) ThisThread::sleep_for(1s); }
    if (resolver.AddFullyConnected() != kTfLiteOk) { print_str("AddFullyConnected failed\r\n"); while (true) ThisThread::sleep_for(1s); }
    if (resolver.AddDequantize() != kTfLiteOk)     { print_str("AddDequantize failed\r\n"); while (true) ThisThread::sleep_for(1s); }

    // ===== Interpreter =====
    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

    TfLiteStatus alloc_status = interpreter.AllocateTensors();
    if (alloc_status != kTfLiteOk) {
        print_str("ERROR: AllocateTensors() failed\r\n");
        while (true) ThisThread::sleep_for(1s);
    }

    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    if (!input || !output) {
        print_str("ERROR: input/output tensor null\r\n");
        while (true) ThisThread::sleep_for(1s);
    }

    // ===== Sanity check =====
    // Bu modelde input int8 ve 10 feature bekliyoruz.
    // Eğer farklıysa burada yakalayıp düzeltirsin.
    if (input->type != kTfLiteInt8) {
        print_str("ERROR: input tensor is not int8\r\n");
        while (true) ThisThread::sleep_for(1s);
    }
    if (output->type != kTfLiteInt8) {
        print_str("ERROR: output tensor is not int8\r\n");
        while (true) ThisThread::sleep_for(1s);
    }

    print_str("TFLite Micro ready\r\n");

    // ===== Example feature vector (Colab first row) =====
    float features[10] = {
        1.144620f,   // ax_mean
        9.883776f,   // ay_mean
        0.022133f,   // az_mean
        1.497014f,   // ax_std
        2.104049f,   // ay_std
        1.698891f,   // az_std
        10.188327f,  // mag_mean
        2.179899f,   // mag_std
        3.851911f,   // mag_min
        17.060185f   // mag_max
    };

    // ===== Quantize input =====
    const float in_scale = input->params.scale;
    const int   in_zero  = input->params.zero_point;

    // input->bytes 10 olmalı (10 int8)
    for (int i = 0; i < 10; i++) {
        int q = (int)lrintf(features[i] / in_scale) + in_zero;
        if (q > 127)  q = 127;
        if (q < -128) q = -128;
        input->data.int8[i] = (int8_t)q;
    }

    // ===== Invoke =====
    if (interpreter.Invoke() != kTfLiteOk) {
        print_str("ERROR: Invoke() failed\r\n");
        while (true) ThisThread::sleep_for(1s);
    }

    // ===== Dequantize output =====
    const float out_scale = output->params.scale;
    const int   out_zero  = output->params.zero_point;

    int8_t y_q = output->data.int8[0];
    float  y   = (float)(y_q - out_zero) * out_scale;

    print_float("Output probability: ", y);

    // (Senin önceki mantığın)
    if (y > 0.5f) {
        print_str("Prediction: NOT WALKING\r\n");
    } else {
        print_str("Prediction: WALKING\r\n");
    }

    while (true) {
        ThisThread::sleep_for(1s);
    }
}
