#include "mbed.h"
#include "arm_math.h"
#include <cstdio>
#include <cstdlib>

// -------------------------
//  Kullanıcı ayarları
// -------------------------
#define ARR_SIZE 64  // FFT boyutu (2^n olmalı)
#define NUM_AXES 3   // x, y, z

// -------------------------
//  Örnek veri üretici fonksiyon (sensör yoksa test için)
// -------------------------
float** get_data(int arr_size) {
    float** data = (float**)malloc(NUM_AXES * sizeof(float*));
    for (int i = 0; i < NUM_AXES; i++) {
        data[i] = (float*)malloc(arr_size * sizeof(float));
        for (int j = 0; j < arr_size; j++) {
            // Basit sinüs dalgası (örnek veriler)
            data[i][j] = 0.5f * arm_sin_f32(2 * PI * j / arr_size * (i + 1));
        }
    }
    return data;
}

// -------------------------
//  Ana program
// -------------------------
int main() {
    printf("\n=== FFT Feature Extraction Başlatılıyor ===\n");

    int arr_size = ARR_SIZE;
    float **acc_data = get_data(arr_size);

    float fft_output[NUM_AXES][ARR_SIZE];
    float fft_std[NUM_AXES];
    float fft_abs[NUM_AXES][ARR_SIZE];

    float sma_x = 0, sma_y = 0, sma_z = 0, sma;
    float x_mean = 0, y_mean = 0, z_mean = 0;
    int x_pos = 0, y_pos = 0, z_pos = 0;

    // -------------------------
    // Ortalama ve pozitif sayımı
    // -------------------------
    for (int i = 0; i < arr_size; i++) {
        x_mean += acc_data[0][i];
        y_mean += acc_data[1][i];
        z_mean += acc_data[2][i];

        x_pos += (acc_data[0][i] > 0);
        y_pos += (acc_data[1][i] > 0);
        z_pos += (acc_data[2][i] > 0);
    }

    x_mean /= arr_size;
    y_mean /= arr_size;
    z_mean /= arr_size;

    // -------------------------
    // FFT hesaplaması
    // -------------------------
    arm_rfft_fast_instance_f32 fft;
    arm_status res = arm_rfft_fast_init_f32(&fft, arr_size);
    if (res != ARM_MATH_SUCCESS) {
        printf("FFT init failed!\n");
        return 1;
    }

    for (int i = 0; i < NUM_AXES; i++) {
        arm_rfft_fast_f32(&fft, acc_data[i], fft_output[i], 0);
        arm_std_f32(fft_output[i], arr_size / 2, &fft_std[i]);
        arm_abs_f32(fft_output[i], fft_abs[i], arr_size);
    }

    // -------------------------
    // SMA hesaplama
    // -------------------------
    for (int i = 0; i < arr_size; i++) {
        sma_x += fft_abs[0][i];
        sma_y += fft_abs[1][i];
        sma_z += fft_abs[2][i];
    }
    sma = (sma_x + sma_y + sma_z) / arr_size;

    // -------------------------
    // Sonuçları yazdır
    // -------------------------
    printf("\n--- Özellikler ---\n");
    printf("x_mean = %.5f, y_mean = %.5f, z_mean = %.5f\n", x_mean, y_mean, z_mean);
    printf("x_pos = %d, y_pos = %d, z_pos = %d\n", x_pos, y_pos, z_pos);
    printf("x_fft_std = %.5f, y_fft_std = %.5f, z_fft_std = %.5f\n",
           fft_std[0], fft_std[1], fft_std[2]);
    printf("SMA (Signal Magnitude Area) = %.5f\n", sma);
    printf("===============================\n");

    // Bellek temizliği
    for (int i = 0; i < NUM_AXES; i++) free(acc_data[i]);
    free(acc_data);

    while (1) {
        thread_sleep_for(1000);
    }
}