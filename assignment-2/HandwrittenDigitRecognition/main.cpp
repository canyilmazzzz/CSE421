#include "hdr_dt_config.h"
#include "mbed.h"

extern const int LEFT_CHILDREN[NUM_NODES];
extern const int RIGHT_CHILDREN[NUM_NODES];
extern const int SPLIT_FEATURE[NUM_NODES];
extern const float THRESHOLDS[NUM_NODES];
extern const int VALUES[NUM_NODES][NUM_CLASSES];

int predict_digit(const float features[NUM_FEATURES]) {
    int node = 0;

    while (true) {
        int left = LEFT_CHILDREN[node];
        int right = RIGHT_CHILDREN[node];

        if (left == -1 && right == -1) {
            int best_class = 0;
            int best_value = VALUES[node][0];

            for (int c = 1; c < NUM_CLASSES; c++) {
                if (VALUES[node][c] > best_value) {
                    best_value = VALUES[node][c];
                    best_class = c;
                }
            }
            return best_class;
        }

        int feature_index = SPLIT_FEATURE[node];
        float threshold = THRESHOLDS[node];

        if (features[feature_index] <= threshold)
            node = left;
        else
            node = right;
    }
}

int main() {
    float input_features[NUM_FEATURES] = {
        0.0023,     0.0000012,     0.00032,      -0.0000043,
        0.00000052, -0.0000000087, 0.00000000012};

    int pred = predict_digit(input_features);

    printf("Predicted digit: %d\n", pred);

    while (true) {
        thread_sleep_for(1000);
    }
}
