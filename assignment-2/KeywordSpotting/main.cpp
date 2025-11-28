#include "kws_knn_config.h"
#include "mbed.h"

int knn_predict(const float input[NUM_FEATURES]) {
    int neighbor_labels[NUM_NEIGHBORS];
    float neighbor_distances[NUM_NEIGHBORS];

    for (int i = 0; i < NUM_NEIGHBORS; i++) {
        neighbor_distances[i] = 1e30f;
        neighbor_labels[i] = -1;
    }

    for (int i = 0; i < NUM_SAMPLES; i++) {
        float dist = 0.0f;
        for (int j = 0; j < NUM_FEATURES; j++) {
            float diff = input[j] - DATA[i][j];
            dist += diff * diff;
        }

        for (int k = 0; k < NUM_NEIGHBORS; k++) {
            if (dist < neighbor_distances[k]) {
                for (int s = NUM_NEIGHBORS - 1; s > k; s--) {
                    neighbor_distances[s] = neighbor_distances[s - 1];
                    neighbor_labels[s] = neighbor_labels[s - 1];
                }
                neighbor_distances[k] = dist;
                neighbor_labels[k] = DATA_LABELS[i];
                break;
            }
        }
    }

    int votes[NUM_CLASSES] = {0};
    for (int i = 0; i < NUM_NEIGHBORS; i++) {
        votes[neighbor_labels[i]]++;
    }

    int max_votes = -1;
    int predicted_label = -1;
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            predicted_label = i;
        }
    }

    return predicted_label;
}

int main() {
    printf("Starting KWS KNN demo...\n");

    float example_input[NUM_FEATURES] = {0};

    int predicted = knn_predict(example_input);
    printf("Predicted class: %d (%s)\n", predicted, LABELS[predicted]);

    while (true) {
    }
}
